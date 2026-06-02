from src.utils import ensure_dir, ensure_file, get_logger, timer


def test_get_logger_returns_logger():
    logger = get_logger("test")
    assert logger.name == "test"


def test_get_logger_same_instance():
    first = get_logger("same")
    second = get_logger("same")
    assert first is second


def test_ensure_file_existing(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("ok", encoding="utf-8")
    assert ensure_file(file_path) == file_path


def test_ensure_file_missing_raises(tmp_path):
    missing = tmp_path / "missing.txt"
    try:
        ensure_file(missing)
    except FileNotFoundError as exc:
        assert "Expected file not found" in str(exc)
    else:
        raise AssertionError("FileNotFoundError was not raised")


def test_ensure_dir_creates(tmp_path):
    directory = tmp_path / "nested" / "dir"
    result = ensure_dir(directory)
    assert result == directory
    assert directory.exists()
    assert directory.is_dir()


def test_timer_runs_function(capsys):
    @timer
    def sample(value):
        return value + 1

    assert sample(1) == 2
    captured = capsys.readouterr()
    assert "[timer] sample finished in" in captured.out
