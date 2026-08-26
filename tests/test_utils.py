from src.utils import check_mismatch


def test_check_mismatch_reports_different_files(tmp_path):
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output.txt"
    input_path.write_text("expected", encoding="utf-8")
    output_path.write_text("actual", encoding="utf-8")

    assert check_mismatch(input_path, output_path=output_path) is False


def test_check_mismatch_reports_identical_files(tmp_path):
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output.txt"
    input_path.write_text("same", encoding="utf-8")
    output_path.write_text("same", encoding="utf-8")

    assert check_mismatch(input_path, output_path=output_path) is True
