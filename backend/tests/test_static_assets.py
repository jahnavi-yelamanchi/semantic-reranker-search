from pathlib import Path

from app import main


def test_static_dir_points_inside_backend_package():
    expected = Path(main.__file__).resolve().parents[1] / "frontend_dist"
    assert main.static_dir == expected
