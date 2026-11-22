#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Field { const char* data; std::size_t size; };

template <std::size_t N>
bool split_csv(const std::string& line, std::array<Field, N>& fields) {
    std::size_t field = 0, start = 0;
    for (std::size_t i = 0; i <= line.size(); ++i) {
        if (i == line.size() || line[i] == ',') {
            if (field == N) return false;
            fields[field++] = Field{line.data() + start, i - start};
            start = i + 1;
        }
    }
    return field == N;
}

double as_double(Field value, const std::string& path,
                 std::size_t row, const char* field) {
    std::size_t i = 0;
    bool negative = false;
    if (i < value.size && (value.data[i] == '-' || value.data[i] == '+')) negative = value.data[i++] == '-';
    double parsed = 0.0;
    std::size_t digits = 0;
    while (i < value.size && value.data[i] >= '0' && value.data[i] <= '9') {
        parsed = parsed * 10.0 + static_cast<double>(value.data[i++] - '0');
        ++digits;
    }
    if (i < value.size && value.data[i] == '.') {
        ++i;
        double place = 0.1;
        while (i < value.size && value.data[i] >= '0' && value.data[i] <= '9') {
            parsed += static_cast<double>(value.data[i++] - '0') * place;
            place *= 0.1;
            ++digits;
        }
    }
    int exponent = 0;
    if (i < value.size && (value.data[i] == 'e' || value.data[i] == 'E')) {
        ++i;
        bool exponent_negative = false;
        if (i < value.size && (value.data[i] == '-' || value.data[i] == '+')) exponent_negative = value.data[i++] == '-';
        std::size_t exponent_digits = 0;
        while (i < value.size && value.data[i] >= '0' && value.data[i] <= '9') {
            exponent = exponent * 10 + (value.data[i++] - '0');
            ++exponent_digits;
        }
        if (!exponent_digits || exponent > 308)
            throw std::runtime_error(path + ": invalid " + field + " at row " + std::to_string(row));
        if (exponent_negative) exponent = -exponent;
    }
    if (!digits || i != value.size)
        throw std::runtime_error(path + ": invalid " + field + " at row " + std::to_string(row));
    if (exponent) parsed *= std::pow(10.0, exponent);
    if (negative) parsed = -parsed;
    if (!std::isfinite(parsed))
        throw std::runtime_error(path + ": invalid " + field + " at row " + std::to_string(row));
    return parsed;
}

double queue_imbalance(double bid, double ask) {
    const double total = bid + ask;
    return total > 0.0 ? (bid - ask) / total : 0.0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: lobster_features MESSAGE.csv ORDERBOOK.csv OUTPUT.csv\n";
        return 2;
    }
    const std::string message_path = argv[1];
    const std::string book_path = argv[2];
    const std::string output_path = argv[3];
    std::ifstream messages(message_path), books(book_path);
    std::ofstream output(output_path, std::ios::binary);
    if (!messages || !books || !output) {
        std::cerr << "could not open an input or output file\n";
        return 2;
    }

    std::string message_line, book_line;
    double previous_time = -1.0;
    double previous_bid_price = 0.0, previous_bid_size = 0.0;
    double previous_ask_price = 0.0, previous_ask_size = 0.0;
    std::uint64_t rows = 0;
    const auto started = std::chrono::steady_clock::now();
    static_assert(sizeof(double) == 8, "binary feature output requires 64-bit doubles");
    std::vector<double> output_buffer;
    output_buffer.reserve((1 << 20) / sizeof(double));

    while (true) {
        const bool has_message = static_cast<bool>(std::getline(messages, message_line));
        const bool has_book = static_cast<bool>(std::getline(books, book_line));
        if (!has_message && !has_book) break;
        if (has_message != has_book) throw std::runtime_error("message/orderbook row counts differ");
        std::array<Field, 6> m{};
        std::array<Field, 20> b{};
        if (!split_csv(message_line, m) || !split_csv(book_line, b))
            throw std::runtime_error("expected six message fields and twenty level-5 book fields");

        const double time = as_double(m[0], message_path, rows + 1, "time");
        const double raw_event_type = as_double(m[1], message_path, rows + 1, "event type");
        const double raw_order_id = as_double(m[2], message_path, rows + 1, "order id");
        const double raw_event_price = as_double(m[4], message_path, rows + 1, "event price");
        const double raw_direction = as_double(m[5], message_path, rows + 1, "direction");
        if (std::floor(raw_event_type) != raw_event_type || std::floor(raw_direction) != raw_direction ||
            std::floor(raw_order_id) != raw_order_id || std::floor(raw_event_price) != raw_event_price)
            throw std::runtime_error("non-integer event type, order id, price, or direction at row " + std::to_string(rows + 1));
        const int event_type = static_cast<int>(raw_event_type);
        const double size = as_double(m[3], message_path, rows + 1, "size");
        const int direction = static_cast<int>(raw_direction);
        const bool known_event = event_type >= 1 && event_type <= 7;
        const bool valid_event_price = event_type == 7 ? raw_event_price >= -1 && raw_event_price <= 1 : raw_event_price > 0;
        if (time < previous_time || size < 0 || raw_order_id < 0 || !known_event || !valid_event_price ||
            (direction != -1 && direction != 1))
            throw std::runtime_error("non-monotone time or invalid event fields at row " + std::to_string(rows + 1));
        previous_time = time;

        std::array<double, 5> ask_price{}, ask_size{}, bid_price{}, bid_size{};
        for (std::size_t level = 0; level < 5; ++level) {
            ask_price[level] = as_double(b[level * 4], book_path, rows + 1, "ask price");
            ask_size[level] = as_double(b[level * 4 + 1], book_path, rows + 1, "ask size");
            bid_price[level] = as_double(b[level * 4 + 2], book_path, rows + 1, "bid price");
            bid_size[level] = as_double(b[level * 4 + 3], book_path, rows + 1, "bid size");
            if (ask_size[level] < 0 || bid_size[level] < 0 ||
                (ask_size[level] > 0 && ask_price[level] <= 0) ||
                (bid_size[level] > 0 && bid_price[level] <= 0))
                throw std::runtime_error("invalid occupied book price or negative size at row " + std::to_string(rows + 1));
        }
        if (ask_price[0] <= 0 || bid_price[0] <= 0 || bid_price[0] > ask_price[0])
            throw std::runtime_error("invalid or crossed top of book at row " + std::to_string(rows + 1));

        double bid_depth = 0.0, ask_depth = 0.0;
        for (std::size_t level = 0; level < 5; ++level) {
            bid_depth += bid_size[level];
            ask_depth += ask_size[level];
        }
        double signed_event_size = 0.0;
        if (event_type == 1) signed_event_size = direction * size;
        else if (event_type == 2 || event_type == 3 || event_type == 4)
            signed_event_size = -direction * size;

        double ofi = 0.0;
        if (rows > 0) {
            if (bid_price[0] > previous_bid_price) ofi += bid_size[0];
            else if (bid_price[0] == previous_bid_price) ofi += bid_size[0] - previous_bid_size;
            else ofi -= previous_bid_size;
            if (ask_price[0] < previous_ask_price) ofi -= ask_size[0];
            else if (ask_price[0] == previous_ask_price) ofi -= ask_size[0] - previous_ask_size;
            else ofi += previous_ask_size;
        }

        std::array<double, 36> record{
            time, static_cast<double>(event_type), size, static_cast<double>(direction),
            signed_event_size, (ask_price[0] + bid_price[0]) / 2.0,
            ask_price[0] - bid_price[0], bid_price[0], ask_price[0],
            bid_size[0], ask_size[0], queue_imbalance(bid_size[0], ask_size[0]),
            queue_imbalance(bid_depth, ask_depth), ofi, raw_order_id, raw_event_price,
        };
        for (std::size_t level = 0; level < 5; ++level) {
            const std::size_t offset = 16 + level * 4;
            record[offset] = ask_price[level];
            record[offset + 1] = ask_size[level];
            record[offset + 2] = bid_price[level];
            record[offset + 3] = bid_size[level];
        }
        output_buffer.insert(output_buffer.end(), record.begin(), record.end());
        if (output_buffer.size() >= (1 << 20) / sizeof(double)) {
            output.write(reinterpret_cast<const char*>(output_buffer.data()),
                         static_cast<std::streamsize>(output_buffer.size() * sizeof(double)));
            output_buffer.clear();
        }

        previous_bid_price = bid_price[0]; previous_bid_size = bid_size[0];
        previous_ask_price = ask_price[0]; previous_ask_size = ask_size[0];
        ++rows;
    }
    if (!output_buffer.empty())
        output.write(reinterpret_cast<const char*>(output_buffer.data()),
                     static_cast<std::streamsize>(output_buffer.size() * sizeof(double)));
    if (!output) throw std::runtime_error("failed while writing feature output");
    output.close();
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::cerr << "processed " << rows << " aligned events in " << elapsed << " seconds ("
              << (elapsed > 0 ? rows / elapsed : 0.0) << " events/s)\n";
    return 0;
}
