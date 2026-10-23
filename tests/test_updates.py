"""
Updates: version checks, picking the right release file, downloading
(with split parts and SHA256SUMS), the "Update available" window, the
daily check, and the release-tag guard used by the Build workflow.
No real network: GitHub is replaced by a fake, downloads by a local
HTTP server.
"""
import hashlib
import http.server
import threading
import time
from functools import partial
from pathlib import Path

import pytest

from services import model_store, updates

# The file names of the real v2.3.0 release (built while version.py said 2.2.1).
V230_ASSETS = [
    "SegmentME-2.2.1-linux-cpu-arm64.tar.gz",
    "SegmentME-2.2.1-linux-cpu-x64.tar.gz",
    "SegmentME-2.2.1-linux-cuda-x64.tar.gz.part-00",
    "SegmentME-2.2.1-linux-cuda-x64.tar.gz.part-01",
    "SegmentME-2.2.1-macos-cpu-arm64.dmg",
    "segmentme-cuda_2.2.1_amd64.deb.part-00",
    "segmentme-cuda_2.2.1_amd64.deb.part-01",
    "SegmentME-Setup-2.2.1-windows-cpu-x64.exe",
    "SegmentME-Setup-2.2.1-windows-cuda-x64.exe",
    "segmentme_2.2.1_amd64.deb",
    "segmentme_2.2.1_arm64.deb",
    "SHA256SUMS",
]


def github_answer(tag="v2.4.0", names=V230_ASSETS, base="https://example.invalid/dl"):
    return {
        "tag_name": tag,
        "body": "## What's changed\n* Faster things",
        "html_url": f"https://github.com/{updates.REPOSITORY}/releases/tag/{tag}",
        "assets": [{"name": n, "size": 10 + i, "browser_download_url": f"{base}/{n}"}
                   for i, n in enumerate(names)],
    }


def inst(system, arch, flavor, kind):
    return updates.Installation(system, arch, flavor, kind)


# ---------- versions ----------
@pytest.mark.parametrize("text, expected", [
    ("v2.3.0", (2, 3, 0)), ("2.3.1", (2, 3, 1)), ("v3", (3, 0, 0)), ("2.10", (2, 10, 0)),
    ("", None), ("latest", None), ("v2.3.0-beta", None), (None, None),
])
def test_parse_version(text, expected):
    assert updates.parse_version(text) == expected


def test_is_newer_compares_numbers_not_text():
    assert updates.is_newer("2.10.0", "2.9.0")
    assert updates.is_newer("v2.3.1", "2.3.0")
    assert not updates.is_newer("2.3.0", "2.3.0")
    assert not updates.is_newer("2.2.9", "2.3.0")
    assert not updates.is_newer("nonsense", "2.3.0")


def test_release_from_github():
    release = updates.Release.from_github(github_answer("v2.4.0"))
    assert release.version == "2.4.0" and release.tag == "v2.4.0"
    assert "Faster" in release.notes
    assert len(release.assets) == len(V230_ASSETS)
    with pytest.raises(updates.UpdateError):
        updates.Release.from_github({"tag_name": "nightly"})


def test_running_from_source_is_detected():
    assert updates.current_installation().kind == updates.SOURCE


# ---------- which file ----------
@pytest.mark.parametrize("installation, name, parts", [
    (inst("windows", "x64", "cpu", updates.WINDOWS_INSTALLER), "SegmentME-Setup-2.2.1-windows-cpu-x64.exe", 1),
    (inst("windows", "x64", "cuda", updates.WINDOWS_INSTALLER), "SegmentME-Setup-2.2.1-windows-cuda-x64.exe", 1),
    (inst("macos", "arm64", "cpu", updates.MACOS_DMG), "SegmentME-2.2.1-macos-cpu-arm64.dmg", 1),
    (inst("linux", "x64", "cpu", updates.LINUX_DEB), "segmentme_2.2.1_amd64.deb", 1),
    (inst("linux", "arm64", "cpu", updates.LINUX_DEB), "segmentme_2.2.1_arm64.deb", 1),
    (inst("linux", "x64", "cuda", updates.LINUX_DEB), "segmentme-cuda_2.2.1_amd64.deb", 2),
    (inst("linux", "x64", "cpu", updates.LINUX_TARBALL), "SegmentME-2.2.1-linux-cpu-x64.tar.gz", 1),
    (inst("linux", "x64", "cuda", updates.LINUX_TARBALL), "SegmentME-2.2.1-linux-cuda-x64.tar.gz", 2),
])
def test_pick_download_matches_each_build(installation, name, parts):
    release = updates.Release.from_github(github_answer())
    item = updates.pick_download(release, installation)
    assert item.name == name
    assert len(item.parts) == parts
    if parts > 1:   # in order
        assert [p.name for p in item.parts] == [f"{name}.part-00", f"{name}.part-01"]
    assert item.checksums.name == "SHA256SUMS"
    # CPU never gets the CUDA file (or the other way round)
    assert ("cuda" in item.name) == (installation.flavor == "cuda")


def test_pick_download_none_when_nothing_fits():
    release = updates.Release.from_github(github_answer())
    assert updates.pick_download(release, inst("macos", "x64", "cpu", updates.MACOS_DMG)) is None
    assert updates.pick_download(release, inst("linux", "x64", "cpu", updates.SOURCE)) is None


# ---------- downloading ----------
@pytest.fixture
def server(tmp_path):
    """A local web server for the files in tmp_path/'served'."""
    served = tmp_path / "served"
    served.mkdir()

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(served)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield served, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _publish(served, base, name, data, pieces=1, sums=True, wrong_sum=False):
    """Put `name` on the server whole or in `pieces`; returns the Download."""
    if pieces == 1:
        (served / name).write_bytes(data)
        parts = [updates.Asset(name, len(data), f"{base}/{name}")]
    else:
        step = -(-len(data) // pieces)
        parts = []
        for i in range(pieces):
            piece = data[i * step:(i + 1) * step]
            pname = f"{name}.part-{i:02d}"
            (served / pname).write_bytes(piece)
            parts.append(updates.Asset(pname, len(piece), f"{base}/{pname}"))
    checksums = None
    if sums:
        digest = hashlib.sha256(b"other" if wrong_sum else data).hexdigest()
        (served / "SHA256SUMS").write_text(f"{'0' * 64}  something-else.exe\n{digest}  {name}\n")
        checksums = updates.Asset("SHA256SUMS", 0, f"{base}/SHA256SUMS")
    return updates.Download(name, parts, checksums)


def test_download_single_file_with_checksum(server, tmp_path):
    served, base = server
    data = bytes(range(256)) * 400
    item = _publish(served, base, "SegmentME-Setup-2.4.0-windows-cpu-x64.exe", data)
    seen = []
    path = updates.download(item, tmp_path / "dl", lambda d, t, text: seen.append((d, t, text)))
    assert path.read_bytes() == data
    assert seen[-1][0] == seen[-1][1] == len(data)
    assert not list((tmp_path / "dl").glob("*.part*"))


def test_download_joins_parts(server, tmp_path):
    served, base = server
    data = b"0123456789" * 5000 + b"end"
    item = _publish(served, base, "segmentme-cuda_2.4.0_amd64.deb", data, pieces=3)
    texts = []
    path = updates.download(item, tmp_path / "dl", lambda d, t, text: texts.append(text))
    assert path.read_bytes() == data
    assert any("part 3 of 3" in t for t in texts)
    assert sorted(p.name for p in (tmp_path / "dl").iterdir()) == [item.name]   # pieces removed


def test_download_with_wrong_checksum_is_deleted(server, tmp_path):
    served, base = server
    item = _publish(served, base, "SegmentME-2.4.0-macos-cpu-arm64.dmg", b"x" * 1000, wrong_sum=True)
    with pytest.raises(updates.UpdateError, match="damaged"):
        updates.download(item, tmp_path / "dl")
    assert not (tmp_path / "dl" / item.name).exists()


def test_download_can_be_cancelled_and_resumed(server, tmp_path):
    served, base = server
    data = b"y" * (3 << 20)
    item = _publish(served, base, "SegmentME-2.4.0-linux-cpu-x64.tar.gz", data)
    calls = {"n": 0}

    def cancel_soon():
        calls["n"] += 1
        return calls["n"] > 2
    with pytest.raises(model_store.DownloadCancelled):
        updates.download(item, tmp_path / "dl", should_cancel=cancel_soon)
    assert not (tmp_path / "dl" / item.name).exists()
    assert updates.download(item, tmp_path / "dl").read_bytes() == data


def test_download_error_becomes_update_error(server, tmp_path):
    _served, base = server
    item = updates.Download("missing.exe", [updates.Asset("missing.exe", 5, f"{base}/missing.exe")])
    with pytest.raises(updates.UpdateError, match="404"):
        updates.download(item, tmp_path / "dl")


def test_fetch_latest_reports_no_connection(monkeypatch):
    import urllib.error

    def offline(*args, **kwargs):
        raise urllib.error.URLError("no route to host")
    monkeypatch.setattr(updates.urllib.request, "urlopen", offline)
    with pytest.raises(updates.UpdateError, match="internet"):
        updates.fetch_latest()


def test_rate_limit_falls_back_to_web_pages(monkeypatch):
    """GitHub's API allows 60 questions an hour per network; then the
    release page and its file list are used instead."""
    import urllib.error

    def limited(*args, **kwargs):
        raise urllib.error.HTTPError(updates.LATEST_URL, 403, "rate limit exceeded", {}, None)
    monkeypatch.setattr(updates.urllib.request, "urlopen", limited)
    monkeypatch.setattr(updates, "fetch_latest_from_web",
                        lambda timeout=15: updates.Release.from_github(github_answer("v7.0.0")))
    assert updates.fetch_latest().version == "7.0.0"

    def also_broken(timeout=15):
        raise OSError("offline")
    monkeypatch.setattr(updates, "fetch_latest_from_web", also_broken)
    with pytest.raises(updates.UpdateError, match="limiting"):
        updates.fetch_latest()


def test_fetch_latest_from_web_reads_redirect_and_file_list(monkeypatch):
    class Redirected:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def geturl(self):
            return f"https://github.com/{updates.REPOSITORY}/releases/tag/v2.5.0"
    monkeypatch.setattr(updates.urllib.request, "urlopen", lambda *a, **k: Redirected())
    prefix = f"/{updates.REPOSITORY}/releases/download/v2.5.0/"
    html = "".join(f'<a href="{prefix}{n}" rel="nofollow">{n}</a>' for n in V230_ASSETS)
    html += '<a href="/other/link">x</a>' + f'<a href="{prefix}SHA256SUMS">again</a>'
    monkeypatch.setattr(updates, "_read_text", lambda url, context, timeout=30: html)
    release = updates.fetch_latest_from_web()
    assert release.version == "2.5.0"
    assert [a.name for a in release.assets] == V230_ASSETS        # no duplicates, no other links
    item = updates.pick_download(release, inst("linux", "x64", "cuda", updates.LINUX_DEB))
    assert item.name == "segmentme-cuda_2.2.1_amd64.deb" and len(item.parts) == 2
    assert item.parts[0].url == f"https://github.com{prefix}segmentme-cuda_2.2.1_amd64.deb.part-00"


def test_download_with_unknown_sizes(server, tmp_path):
    """Files found without the API have no size: progress uses the server's."""
    served, base = server
    data = b"z" * 50000
    item = _publish(served, base, "SegmentME-2.4.0-linux-cpu-x64.tar.gz", data, pieces=2)
    for part in item.parts:
        part.size = 0
    seen = []
    path = updates.download(item, tmp_path / "dl", lambda d, t, text: seen.append((d, t)))
    assert path.read_bytes() == data
    assert seen[-1] == (len(data), len(data))


def test_notes_drop_the_manual_join_section():
    notes = ("### Linux CUDA build (NVIDIA GPUs)\n\nGitHub limits each release file...\n"
             "```bash\ncat a.part-* > a\n```\n\n## What's Changed\n* Class colours\n\n"
             "**Full Changelog**: https://x/compare/v1...v2")
    text = updates.notes_for_app(notes, "https://x/releases/tag/v2")
    assert "part" not in text and "Class colours" in text and "Full Changelog" in text
    assert text.endswith("[Release page on GitHub](https://x/releases/tag/v2)")
    only_join = "### Linux CUDA build\ncat ...\n**Full Changelog**: https://x"
    assert updates.notes_for_app(only_join).startswith("**Full Changelog**")


# ---------- the window ----------
@pytest.fixture
def updater(qapp, tmp_path, monkeypatch, no_popups):
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QMainWindow
    from ui.dialogs import update_dialog
    window = QMainWindow()
    u = update_dialog.Updater(window)
    u.settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    offered = []
    monkeypatch.setattr(u, "offer", lambda release: offered.append(release.version))
    u.offered = offered
    u.popups = no_popups
    yield u
    window.deleteLater()


def _answer_with(monkeypatch, tag=None, error=None):
    def fake():
        if error:
            raise error
        return updates.Release.from_github(github_answer(tag))
    monkeypatch.setattr(updates, "fetch_latest", fake)


def _wait(updater, timeout=5):
    from PyQt6.QtWidgets import QApplication
    end = time.time() + timeout
    while updater.checking and time.time() < end:
        QApplication.processEvents()
        time.sleep(0.005)


def test_manual_check_offers_newer_version(updater, monkeypatch):
    _answer_with(monkeypatch, "v99.0.0")
    assert updater.check_now()
    _wait(updater)
    assert updater.offered == ["99.0.0"]


def test_manual_check_when_up_to_date(updater, monkeypatch):
    from version import __version__
    _answer_with(monkeypatch, f"v{__version__}")
    updater.check_now()
    _wait(updater)
    assert updater.offered == []
    assert updater.popups[-1][0] == "information" and "latest version" in updater.popups[-1][2]


def test_manual_check_shows_error(updater, monkeypatch):
    _answer_with(monkeypatch, error=updates.UpdateError("Could not reach GitHub"))
    updater.check_now()
    _wait(updater)
    assert updater.popups[-1][0] == "warning" and "Could not reach GitHub" in updater.popups[-1][2]


def test_skipped_version_is_quiet_on_start_but_shown_when_asked(updater, monkeypatch):
    from ui.dialogs import update_dialog
    _answer_with(monkeypatch, "v99.0.0")
    updater.settings.setValue(update_dialog.KEY_SKIPPED, "99.0.0")
    updater.check(manual=False)
    _wait(updater)
    assert updater.offered == []
    updater.check_now()
    _wait(updater)
    assert updater.offered == ["99.0.0"]


def test_automatic_check_errors_stay_silent(updater, monkeypatch):
    _answer_with(monkeypatch, error=updates.UpdateError("offline"))
    updater.check(manual=False)
    _wait(updater)
    assert updater.popups == []


def test_daily_check_rules(updater, monkeypatch):
    from ui.dialogs import update_dialog
    _answer_with(monkeypatch, "v99.0.0")
    # From source: never by itself
    assert updater.check_on_start() is False
    updater.installation = inst("windows", "x64", "cpu", updates.WINDOWS_INSTALLER)
    # Switched off in Settings
    updater.auto_check = False
    assert updater.check_on_start() is False
    updater.auto_check = True
    # Checked an hour ago
    updater.settings.setValue(update_dialog.KEY_LAST, time.time() - 3600)
    assert updater.check_on_start() is False
    # Over a day ago
    updater.settings.setValue(update_dialog.KEY_LAST, time.time() - 2 * 24 * 3600)
    assert updater.check_on_start() is True
    _wait(updater)
    assert updater.offered == [] and "99.0.0" in updater.notice.text()   # no window by itself
    assert time.time() - updater.settings.value(update_dialog.KEY_LAST, type=float) < 60


def test_update_dialog_buttons(qapp):
    from PyQt6.QtWidgets import QWidget
    from ui.dialogs.update_dialog import LATER, SKIP, UPDATE, UpdateDialog
    release = updates.Release.from_github(github_answer("v99.0.0"))
    item = updates.pick_download(release, inst("windows", "x64", "cpu", updates.WINDOWS_INSTALLER))
    parent = QWidget()
    for button, choice in (("update_button", UPDATE), ("later_button", LATER), ("skip_button", SKIP)):
        dialog = UpdateDialog(parent, release, item)
        getattr(dialog, button).click()
        assert dialog.choice == choice
    no_file = UpdateDialog(parent, release, None)
    assert no_file.update_button.text() == "Open Download Page"


def test_offer_skip_and_update(qapp, tmp_path, monkeypatch):
    """Skip remembers the version; Update downloads, then (Windows) starts
    the installer and closes the app."""
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
    from ui.dialogs import update_dialog

    window = QMainWindow()
    u = update_dialog.Updater(window)
    u.settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    u.installation = inst("windows", "x64", "cpu", updates.WINDOWS_INSTALLER)
    release = updates.Release.from_github(github_answer("v99.0.0"))

    choice = {"value": update_dialog.SKIP}
    monkeypatch.setattr(update_dialog.UpdateDialog, "exec",
                        lambda self: setattr(self, "choice", choice["value"]))
    assert u.offer(release) == update_dialog.SKIP
    assert u.skipped_version() == "99.0.0"

    got = tmp_path / "dl" / "SegmentME-Setup-2.2.1-windows-cpu-x64.exe"
    monkeypatch.setattr(update_dialog, "download_folder", lambda: tmp_path / "dl")

    def fake_download(item, folder, progress=None, cancelled=None):
        assert item.name == got.name and Path(folder) == tmp_path / "dl"
        got.parent.mkdir(parents=True, exist_ok=True)
        got.write_bytes(b"installer")
        return got
    monkeypatch.setattr(updates, "download", fake_download)
    launched, closed = [], []
    monkeypatch.setattr(update_dialog, "launch", launched.append)
    monkeypatch.setattr(window, "close", lambda: closed.append(True))
    monkeypatch.setattr(QApplication, "quit", staticmethod(lambda: None))
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(updates, "free_space", lambda folder: 10**12)
    choice["value"] = update_dialog.UPDATE
    assert u.offer(release) == update_dialog.UPDATE
    job = u.downloading
    assert job is not None and job.widget is not None      # progress in the status bar
    job.wait()
    assert u.downloading is None
    assert launched == [got] and closed == [True]
    window.deleteLater()


def test_not_enough_disk_space_is_refused(updater, monkeypatch, tmp_path):
    from ui.dialogs import update_dialog
    release = updates.Release.from_github(github_answer("v99.0.0"))
    item = updates.pick_download(release, inst("linux", "x64", "cuda", updates.LINUX_DEB))
    for p in item.parts:
        p.size = 2_000_000_000
    monkeypatch.setattr(update_dialog, "download_folder", lambda: tmp_path / "dl")
    monkeypatch.setattr(updates, "free_space", lambda folder: 1_000_000_000)
    assert updater.download_and_install(release, item) is None
    assert updater.downloading is None
    kind, _title, text = updater.popups[-1]
    assert kind == "warning" and "not enough free disk space" in text
    # parts need room for the pieces and the joined file
    assert updates.space_needed(item, tmp_path / "dl") == 8_000_000_000


def test_failed_download_reports_and_does_not_crash(updater, monkeypatch, tmp_path):
    from ui.dialogs import update_dialog
    release = updates.Release.from_github(github_answer("v99.0.0"))
    item = updates.pick_download(release, inst("windows", "x64", "cpu", updates.WINDOWS_INSTALLER))
    monkeypatch.setattr(update_dialog, "download_folder", lambda: tmp_path / "dl")
    monkeypatch.setattr(updates, "free_space", lambda folder: 10**12)

    def broken(*args, **kwargs):
        raise updates.UpdateError("The disk is full.")
    monkeypatch.setattr(updates, "download", broken)
    job = updater.download_and_install(release, item)
    job.wait()
    assert updater.downloading is None
    assert any(kind == "warning" and "disk is full" in text for kind, _t, text in updater.popups)


def test_closing_the_window_stops_the_download(updater, monkeypatch, tmp_path):
    from ui.dialogs import update_dialog
    release = updates.Release.from_github(github_answer("v99.0.0"))
    item = updates.pick_download(release, inst("windows", "x64", "cpu", updates.WINDOWS_INSTALLER))
    monkeypatch.setattr(update_dialog, "download_folder", lambda: tmp_path / "dl")
    monkeypatch.setattr(updates, "free_space", lambda folder: 10**12)

    def slow(item, folder, progress, cancelled):
        while not cancelled():
            time.sleep(0.01)
        raise model_store.DownloadCancelled()
    monkeypatch.setattr(updates, "download", slow)
    job = updater.download_and_install(release, item)
    assert updater.check_now() is False               # already downloading
    updater.stop()
    job.wait()
    assert job.cancelled and updater.downloading is None


def test_automatic_check_shows_a_status_bar_button_not_a_window(qapp, tmp_path, monkeypatch):
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QMainWindow
    from ui.dialogs import update_dialog
    window = QMainWindow()
    u = update_dialog.Updater(window)
    u.settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    opened = []
    monkeypatch.setattr(u, "offer", lambda release: opened.append(release.version))
    _answer_with(monkeypatch, "v99.0.0")
    u.check(manual=False)
    _wait(u)
    assert opened == []
    assert u.notice is not None and "99.0.0" in u.notice.text()
    u.notice.click()
    assert opened == ["99.0.0"]
    window.deleteLater()


def test_enter_key_cannot_start_the_download(qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QWidget
    from ui.dialogs.update_dialog import LATER, UpdateDialog
    release = updates.Release.from_github(github_answer("v99.0.0"))
    item = updates.pick_download(release, inst("windows", "x64", "cpu", updates.WINDOWS_INSTALLER))
    parent = QWidget()
    dialog = UpdateDialog(parent, release, item)
    dialog.show()
    QTest.keyClick(dialog, Qt.Key.Key_Return)
    QTest.keyClick(dialog, Qt.Key.Key_Escape)
    assert dialog.choice == LATER


def test_clock_set_back_does_not_stop_the_daily_check(updater, monkeypatch):
    from ui.dialogs import update_dialog
    _answer_with(monkeypatch, "v1.0.0")
    updater.installation = inst("windows", "x64", "cpu", updates.WINDOWS_INSTALLER)
    updater.settings.setValue(update_dialog.KEY_LAST, time.time() + 30 * 24 * 3600)
    assert updater.check_on_start() is True
    _wait(updater)


def test_download_folder_falls_back_when_not_writable(qapp, monkeypatch, tmp_path):
    from ui.dialogs import update_dialog
    blocked = tmp_path / "file-not-folder"
    blocked.write_text("x")
    monkeypatch.setattr(update_dialog.QStandardPaths, "writableLocation",
                        staticmethod(lambda *_: str(blocked / "Downloads")))
    monkeypatch.setattr(model_store, "user_data_dir", lambda: tmp_path / "user")
    assert update_dialog.download_folder() == tmp_path / "user" / "updates"


def test_friendly_errors():
    import errno
    import socket
    assert "disk is full" in updates.friendly_error(OSError(errno.ENOSPC, "No space left"))
    assert "stopped responding" in updates.friendly_error(socket.timeout("timed out"))
    assert "interrupted" in updates.friendly_error(ConnectionResetError())
    assert "not allowed" in updates.friendly_error(PermissionError(13, "denied"), "/x")


def test_old_broken_file_is_downloaded_again(server, tmp_path):
    served, base = server
    data = b"new installer" * 1000
    item = _publish(served, base, "SegmentME-Setup-2.2.1-windows-cpu-x64.exe", data)
    (tmp_path / "dl").mkdir()
    (tmp_path / "dl" / item.name).write_bytes(b"an older file with the same name")
    assert updates.download(item, tmp_path / "dl").read_bytes() == data


def test_pieces_finished_earlier_are_not_downloaded_again(server, tmp_path):
    served, base = server
    data = b"abc" * 30000
    item = _publish(served, base, "segmentme-cuda_2.4.0_amd64.deb", data, pieces=2)
    updates.download(item, tmp_path / "dl")
    # The app closed while joining: both pieces are there, the joined file is not
    (tmp_path / "dl" / item.name).unlink()
    for i, part in enumerate(item.parts):
        (tmp_path / "dl" / part.name).write_bytes(data[i * 45000:(i + 1) * 45000])
        (served / part.name).unlink()                  # the server would now fail
    assert updates.download(item, tmp_path / "dl").read_bytes() == data


def test_window_has_update_menu_items(qapp, tmp_path, no_popups):
    import json
    from PyQt6.QtWidgets import QMenu
    from tests.conftest import make_image
    from ui.main_window import MainApp
    root = tmp_path / "proj"
    (root / ".segmentme").mkdir(parents=True)
    make_image(root / "images" / "a.png")
    (root / "proj.SEproj").write_text(json.dumps({"name": "proj", "images": ["a.png"]}))
    win = MainApp(db_path=str(root / ".segmentme" / "masks.db"))
    try:
        texts = [a.text() for menu in win.menu_bar.findChildren(QMenu)
                 for a in menu.actions()]
        assert "Check for Updates…" in texts and "About SegmentME" in texts
        assert win.menu_bar.auto_update_action.isCheckable()
        assert win.updater.check_on_start() is False     # tests run from source: never online
    finally:
        win.state_manager.db.conn.close()


# ---------- release tag guard ----------
def test_release_tag_must_match_version_py():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "check_release_version",
        Path(__file__).resolve().parent.parent / "scripts" / "check_release_version.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.check("v2.3.1", "2.3.1") is None
    assert "does not match" in module.check("v2.3.0", "2.2.1")
    assert module.main(["x", "v0.0.0"]) == 1
