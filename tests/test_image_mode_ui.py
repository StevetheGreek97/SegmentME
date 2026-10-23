"""Copy / link choice in the dialogs, New Project, and Import Images."""
import json
import os
import pathlib
import sys

import pytest

from services import image_files
from tests.conftest import make_image


def _linked(path, original) -> bool:
    """A link (symbolic or hard) to `original`, not a separate copy."""
    return os.path.samefile(path, original)


# --------------------------------------------------------------------------
# The choice widget
# --------------------------------------------------------------------------
def test_choice_defaults_to_copy_and_remembers_the_last_choice(qapp, monkeypatch):
    from ui.dialogs.image_mode import ImageModeChoice, ImageModeDialog
    monkeypatch.setattr(ImageModeChoice, "_last_mode", image_files.COPY)
    a = ImageModeChoice()
    assert a.mode() == image_files.COPY and a.hint.isHidden()
    a.link_radio.setChecked(True)
    assert a.mode() == image_files.LINK and not a.hint.isHidden()
    b = ImageModeDialog(None, 3)
    assert b.mode() == image_files.LINK   # remembered within the session
    if sys.platform == "win32":
        assert "another drive" in a.hint.text()


# --------------------------------------------------------------------------
# New Project
# --------------------------------------------------------------------------
@pytest.fixture
def originals(tmp_path):
    return [make_image(tmp_path / "originals" / n) for n in ("one.png", "two.jpg")]


@pytest.mark.parametrize("mode", [image_files.COPY, image_files.LINK])
def test_new_project(qapp, tmp_path, originals, no_popups, mode):
    from ui.dialogs.project_dialog import CreateProjectDialog
    d = CreateProjectDialog()
    d.path_input.setText(str(tmp_path))
    d.name_input.setText("proj")
    for f in originals:
        assert d.img_list.add_file(f)
    d.image_mode.set_mode(mode)
    d._finalize()

    images = tmp_path / "proj" / "images"
    assert d.selected_project_db == str(tmp_path / "proj" / ".segmentme" / "masks.db")
    assert sorted(os.listdir(images)) == ["one.png", "two.jpg"]
    seproj = json.loads((tmp_path / "proj" / "proj.SEproj").read_text(encoding="utf-8"))
    assert seproj["images"] == ["one.png", "two.jpg"]
    for f in originals:
        assert _linked(images / f.name, f) == (mode == image_files.LINK)
    assert no_popups == []


def test_new_project_refuses_two_images_with_the_same_name(qapp, tmp_path):
    from ui.dialogs.project_dialog import CreateProjectDialog
    d = CreateProjectDialog()
    assert d.img_list.add_file(make_image(tmp_path / "a" / "scan.png"))
    assert not d.img_list.add_file(make_image(tmp_path / "b" / "SCAN.png"))
    assert d.img_list.count() == 1


def test_new_project_failure_leaves_no_half_project(qapp, tmp_path, originals, no_popups, monkeypatch):
    from ui.dialogs import project_dialog
    d = project_dialog.CreateProjectDialog()
    d.path_input.setText(str(tmp_path))
    d.name_input.setText("proj")
    for f in originals:
        d.img_list.add_file(f)
    calls = []

    real = image_files.place_image

    def fail_second(src, dest, mode, *args):
        calls.append(src)
        if len(calls) == 2:
            raise OSError(28, "No space left on device")
        return real(src, dest, mode, *args)
    monkeypatch.setattr(image_files, "place_image", fail_second)
    d._finalize()
    assert not (tmp_path / "proj").exists()
    assert all(f.exists() for f in originals)
    assert no_popups and no_popups[0][0] == "critical"


def test_add_folder_finds_images_in_subfolders(qapp, tmp_path, no_popups, monkeypatch):
    """Add Folder… lists every image in the folder and its subfolders;
    a second image with an already listed name is left out, and said so."""
    from PyQt6.QtWidgets import QFileDialog
    from ui.dialogs.project_dialog import CreateProjectDialog
    src = tmp_path / "scans"
    for rel in ("week 1/img2.png", "week 1/img10.png", "week 10/b.jpg", "week 2/img2.png",
                "top.TIF", ".hidden/secret.png"):
        make_image(src / rel)
    (src / "week 1" / "notes.txt").write_text("not an image")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(src)))
    d = CreateProjectDialog()
    d._import_folder()
    names = [pathlib.Path(f).name for f in d.img_list.file_list]
    assert names == ["b.jpg", "img2.png", "img10.png", "top.TIF"]
    assert pathlib.Path(d.img_list.file_list[1]).parent.name == "week 1"   # the first one is kept
    assert not d.skipped_note.isHidden() and "img2.png" in d.skipped_note.text()
    # and the project gets them all
    d.path_input.setText(str(tmp_path))
    d.name_input.setText("proj")
    d._finalize()
    assert d.selected_project_db
    assert sorted(p.name for p in (tmp_path / "proj" / "images").iterdir()) == sorted(names)


def test_dropping_a_folder_adds_its_images(qapp, tmp_path):
    from ui.dialogs.project_dialog import DraggableImageList
    make_image(tmp_path / "f" / "sub" / "a.png")
    make_image(tmp_path / "loose.jpg")
    lst = DraggableImageList()
    added, skipped = lst.add_paths([tmp_path / "f", tmp_path / "loose.jpg", tmp_path / "nothing.txt"])
    assert added == 2 and skipped == []
    assert [pathlib.Path(f).name for f in lst.file_list] == ["a.png", "loose.jpg"]


def test_new_project_has_no_annotations_checkbox(qapp):
    from ui.dialogs.project_dialog import CreateProjectDialog
    assert not hasattr(CreateProjectDialog(), "annotations_check")


def test_new_project_refuses_when_copies_do_not_fit(qapp, tmp_path, originals, no_popups, monkeypatch):
    from collections import namedtuple
    from ui.dialogs import project_dialog, place_progress
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(place_progress.shutil, "disk_usage", lambda p: Usage(10, 10, 1000))
    d = project_dialog.CreateProjectDialog()
    d.path_input.setText(str(tmp_path))
    d.name_input.setText("proj")
    for f in originals:
        d.img_list.add_file(f)
    d.image_mode.set_mode(image_files.COPY)
    d._finalize()
    assert not (tmp_path / "proj").exists() and d.selected_project_db is None
    assert no_popups and "only" in no_popups[0][2] and "Link to the original files" in no_popups[0][2]
    # Linking needs no space, so it goes ahead
    d.image_mode.set_mode(image_files.LINK)
    d._finalize()
    assert (tmp_path / "proj" / "images" / "one.png").exists()
