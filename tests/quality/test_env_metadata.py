from __future__ import annotations

from polarism.env_metadata import write_hardware_tex


def test_hardware_tex_written(tmp_path) -> None:
    path = tmp_path / "hardware.tex"
    write_hardware_tex(path)
    contents = path.read_text(encoding="utf-8")
    assert contents
    assert "Python" in contents
    assert "NumPy" in contents
    assert "OS" in contents
