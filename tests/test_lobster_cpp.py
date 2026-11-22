import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


def test_cpp_feature_extractor_checks_alignment_and_emits_top_of_book(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler is not installed")
    project_root = Path(__file__).resolve().parents[1]
    executable = tmp_path / "lobster_features.exe"
    subprocess.run([compiler, "-std=c++17", "-O2",
                    str(project_root / "src" / "lobster_features.cpp"), "-o", str(executable)],
                   check=True)

    message = tmp_path / "message.csv"
    book = tmp_path / "orderbook.csv"
    output = tmp_path / "features.bin"
    message.write_text(
        "34200.0,1,1,10,10000,1\n34200.1,4,1,2,10000,1\n", encoding="ascii")
    first = []
    second = []
    for level in range(5):
        first += [str(10010 + level * 10), "10", str(10000 - level * 10), "5"]
        second += [str(10010 + level * 10), "10", str(10000 - level * 10), "3" if level == 0 else "5"]
    book.write_text(",".join(first) + "\n" + ",".join(second) + "\n", encoding="ascii")
    subprocess.run([str(executable), str(message), str(book), str(output)], check=True,
                   capture_output=True, text=True)

    records = np.fromfile(output, dtype="<f8").reshape(-1, 36)
    assert records.shape == (2, 36)
    assert records[0, 5] == 10005.0
    assert records[0, 6] == 10.0
    assert records[0, 11] == pytest.approx(-1 / 3)
    assert records[0, 13] == 0.0
    assert records[1, 13] == -2.0
    assert records[0, 14] == 1.0
    assert records[0, 15] == 10000.0
    assert records[0, 16:20].tolist() == [10010.0, 10.0, 10000.0, 5.0]


def test_cpp_feature_extractor_rejects_misaligned_rows(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler is not installed")
    project_root = Path(__file__).resolve().parents[1]
    executable = tmp_path / "lobster_features.exe"
    subprocess.run([compiler, "-std=c++17", "-O2",
                    str(project_root / "src" / "lobster_features.cpp"), "-o", str(executable)],
                   check=True)
    message = tmp_path / "message.csv"
    book = tmp_path / "orderbook.csv"
    output = tmp_path / "features.bin"
    message.write_text("34200.0,1,1,10,10000,1\n34200.1,4,1,2,10000,1\n", encoding="ascii")
    valid_book = ",".join(value for level in range(5)
                            for value in (str(10010 + level * 10), "10",
                                          str(10000 - level * 10), "5"))
    book.write_text(valid_book + "\n", encoding="ascii")
    result = subprocess.run([str(executable), str(message), str(book), str(output)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "row counts differ" in result.stderr


@pytest.mark.parametrize("message_line,expected", [
    ("34200.0,9,1,10,10000,1", "invalid event fields"),
    ("34200.0,1,1,-1,10000,1", "invalid event fields"),
    ("34200.0,1,1,10,0,1", "invalid event fields"),
])
def test_cpp_feature_extractor_rejects_invalid_message_fields(tmp_path, message_line, expected):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler is not installed")
    project_root = Path(__file__).resolve().parents[1]
    executable = tmp_path / "lobster_features.exe"
    subprocess.run([compiler, "-std=c++17", "-O2",
                    str(project_root / "src" / "lobster_features.cpp"), "-o", str(executable)],
                   check=True)
    message = tmp_path / "message.csv"
    book = tmp_path / "orderbook.csv"
    output = tmp_path / "features.bin"
    message.write_text(message_line + "\n", encoding="ascii")
    valid_book = ",".join(value for level in range(5)
                            for value in (str(10010 + level * 10), "10",
                                          str(10000 - level * 10), "5"))
    book.write_text(valid_book + "\n", encoding="ascii")
    result = subprocess.run([str(executable), str(message), str(book), str(output)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert expected in result.stderr
