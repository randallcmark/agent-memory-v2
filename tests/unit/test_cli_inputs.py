from io import StringIO

import pytest

from agent_memory_v2.cli_inputs import resolve_text_input


class FakeTTY(StringIO):
    def isatty(self) -> bool:
        return True


class FakePipe(StringIO):
    def isatty(self) -> bool:
        return False


def test_resolve_text_input_prefers_explicit_text():
    assert resolve_text_input(text="hello", text_file=None) == "hello"


def test_resolve_text_input_reads_file(tmp_path):
    path = tmp_path / "input.txt"
    path.write_text("from-file\n", encoding="utf-8")
    assert resolve_text_input(text=None, text_file=str(path)) == "from-file"


def test_resolve_text_input_reads_stdin():
    assert (
        resolve_text_input(text=None, text_file=None, stdin=FakePipe("from-stdin\n"))
        == "from-stdin"
    )


def test_resolve_text_input_rejects_multiple_sources(tmp_path):
    path = tmp_path / "input.txt"
    path.write_text("from-file\n", encoding="utf-8")
    with pytest.raises(ValueError):
        resolve_text_input(text="hello", text_file=str(path))


def test_resolve_text_input_requires_a_source():
    with pytest.raises(ValueError):
        resolve_text_input(text=None, text_file=None, stdin=FakeTTY(""))
