"""Copying or linking images into a project (services/image_files.py)."""
import errno
import os
import shutil
import sys

import pytest

from services import image_files
from services.file_handlers import loader
from tests.conftest import can_symlink, make_image


@pytest.fixture
def src(tmp_path):
    return make_image(tmp_path / "originals" / "scan_01.png")


@pytest.fixture
def images_dir(tmp_path):
    d = tmp_path / "project" / "images"
    d.mkdir(parents=True)
    return d


def test_copy_makes_an_independent_file(src, images_dir):
    dest = images_dir / src.name
    how, reason = image_files.place_image(src, dest, image_files.COPY)
    assert (how, reason) == (image_files.COPIED, None)
    assert dest.read_bytes() == src.read_bytes()
    assert not os.path.islink(dest)
    assert os.stat(dest).st_ino != os.stat(src).st_ino or sys.platform == "win32"
    src.unlink()
    assert dest.exists()


def test_link_uses_the_preferred_link_for_this_os(src, images_dir):
    dest = images_dir / src.name
    how, reason = image_files.place_image(src, dest, image_files.LINK)
    assert reason is None
    if sys.platform == "win32":
        assert how == image_files.HARDLINK      # works for every user on the same drive
        assert os.path.samefile(dest, src) and not os.path.islink(dest)
    else:
        assert how == image_files.SYMLINK
        assert os.path.islink(dest)
        assert os.path.realpath(dest) == os.path.realpath(src)
    assert dest.read_bytes() == src.read_bytes()


def test_link_points_at_the_real_file_not_at_another_link(src, images_dir, tmp_path):
    if not can_symlink(tmp_path):
        pytest.skip("symbolic links not allowed here")
    middle = tmp_path / "middle.png"
    os.symlink(src, middle)
    dest = images_dir / "scan.png"
    image_files.place_image(middle, dest, image_files.LINK)
    middle.unlink()
    assert dest.read_bytes() == src.read_bytes()


def _refuse(winerror=None, err=errno.EXDEV):
    def make(*_args, **_kwargs):
        exc = OSError(err, os.strerror(err))
        if winerror is not None:
            exc.winerror = winerror
        raise exc
    return make


def test_link_falls_back_to_copy_with_a_reason(src, images_dir, monkeypatch):
    monkeypatch.setattr(os, "link", _refuse(err=errno.EXDEV))
    monkeypatch.setattr(os, "symlink", _refuse(err=errno.EPERM))
    dest = images_dir / src.name
    how, reason = image_files.place_image(src, dest, image_files.LINK)
    assert how == image_files.COPIED
    expected = ("the original is on another drive" if sys.platform == "win32"
                else "no permission to make links here")  # first method tried
    assert reason == expected
    assert dest.read_bytes() == src.read_bytes() and not os.path.islink(dest)


def test_windows_order_hard_link_then_symlink(src, images_dir, monkeypatch):
    """Windows: hard link first; other drive -> symbolic link; no rights -> copy."""
    monkeypatch.setattr(sys, "platform", "win32")
    calls = []

    def fake_link(a, b):
        calls.append("hardlink")
        raise OSError(errno.EXDEV, "other drive")

    def fake_symlink(a, b):
        calls.append("symlink")
        exc = OSError(errno.EPERM, "privilege")
        exc.winerror = 1314
        raise exc

    monkeypatch.setattr(os, "link", fake_link)
    monkeypatch.setattr(os, "symlink", fake_symlink)
    how, reason = image_files.place_image(src, images_dir / src.name, image_files.LINK)
    assert calls == ["hardlink", "symlink"]
    assert how == image_files.COPIED and reason == "the original is on another drive"


def test_existing_name_is_never_overwritten(src, images_dir, tmp_path):
    dest = images_dir / src.name
    dest.write_bytes(b"keep me")
    for mode in (image_files.COPY, image_files.LINK):
        with pytest.raises(FileExistsError):
            image_files.place_image(src, dest, mode)
    assert dest.read_bytes() == b"keep me"


def test_broken_link_blocks_the_name_and_nothing_is_written_through_it(src, images_dir, tmp_path):
    if not can_symlink(tmp_path):
        pytest.skip("symbolic links not allowed here")
    gone = tmp_path / "gone" / "scan_01.png"
    dest = images_dir / src.name
    os.symlink(gone, dest)
    with pytest.raises(FileExistsError):
        image_files.place_image(src, dest, image_files.COPY)
    assert not gone.exists()


def test_unknown_mode(src, images_dir):
    with pytest.raises(ValueError):
        image_files.place_image(src, images_dir / src.name, "move")


def test_broken_link_detection_and_message(src, images_dir, tmp_path):
    if not can_symlink(tmp_path):
        pytest.skip("symbolic links not allowed here")
    dest = images_dir / src.name
    image_files.place_image(src, dest, image_files.LINK) if sys.platform != "win32" \
        else os.symlink(src, dest)
    assert not image_files.is_broken_link(dest)
    shutil.move(str(src), str(tmp_path / "moved.png"))
    assert image_files.is_broken_link(dest)
    msg = image_files.missing_image_message(dest)
    assert "original of this linked image is missing" in msg and "scan_01.png" in msg
    # The image list still shows it, so opening the project keeps its masks.
    assert [os.path.basename(p) for p in loader(str(images_dir))] == ["scan_01.png"]


def test_missing_and_unreadable_messages(images_dir):
    assert "missing" in image_files.missing_image_message(images_dir / "nope.png")
    bad = images_dir / "bad.png"
    bad.write_bytes(b"not an image")
    assert "could not be read" in image_files.missing_image_message(bad)


def test_deleting_the_project_keeps_the_originals(src, images_dir):
    image_files.place_image(src, images_dir / src.name, image_files.LINK)
    shutil.rmtree(images_dir.parent)
    assert src.exists() and src.stat().st_size > 0


def test_summarize():
    s = image_files.summarize([(image_files.SYMLINK, None), (image_files.HARDLINK, None),
                               (image_files.COPIED, "the original is on another drive")])
    assert s == ("Images: 2 linked to the original file(s), 1 copied into the project "
                 "(1 could not be linked: the original is on another drive).")
    assert image_files.summarize([(image_files.COPIED, None)]) == "Images: 1 copied into the project."
    assert image_files.summarize([]) == ""


def test_loader_sees_upper_case_extensions(tmp_path):
    for name in ("a.PNG", "b.jpg", "c.TIF", "d.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert sorted(os.path.basename(p) for p in loader(str(tmp_path))) == ["a.PNG", "b.jpg", "c.TIF"]
