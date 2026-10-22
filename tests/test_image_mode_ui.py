"""Copy / link choice in the dialogs, New Project, and Import Images."""
import json
import os
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

    def fail_second(src, dest, mode):
        calls.append(src)
        if len(calls) == 2:
            raise OSError(28, "No space left on device")
        return image_files.place_image(src, dest, mode)
    monkeypatch.setattr(project_dialog.image_files, "place_image", fail_second)
    d._finalize()
    assert not (tmp_path / "proj").exists()
    assert all(f.exists() for f in originals)
    assert no_popups and no_popups[0][0] == "critical"


def test_new_project_can_ask_to_import_annotations(qapp, tmp_path, originals, no_popups):
    from ui.dialogs.project_dialog import CreateProjectDialog
    for checked in (False, True):
        d = CreateProjectDialog()
        d.path_input.setText(str(tmp_path))
        d.name_input.setText(f"proj_{checked}")
        d.img_list.add_file(originals[0])
        d.annotations_check.setChecked(checked)
        d._finalize()
        assert d.selected_project_db and d.import_annotations is checked
