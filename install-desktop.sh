#!/bin/bash
# Register SegmentME with the Linux desktop for the current user:
#
#   * an applications-menu entry (segmentme.desktop)
#   * the MIME type application/x-segmentme-project for *.SEproj files, with
#     SegmentME as its default handler, so double-clicking a project opens it
#   * an icon for .SEproj files
#
# Everything is written under $XDG_DATA_HOME (default ~/.local/share) and
# ~/.config/mimeapps.list; no root needed.
#
#   ./install-desktop.sh              register
#   ./install-desktop.sh --uninstall  remove again
#
# Works both next to the packaged SegmentME executable (the folder unpacked
# from the release archive) and in a source checkout, where it launches
# main.py with .venv/bin/python (override: SEGMENTME_PYTHON=/path/to/python).
# Re-run it after moving the folder.
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
CONFIG="${XDG_CONFIG_HOME:-$HOME/.config}"
MIME_TYPE="application/x-segmentme-project"
DESKTOP_ID="segmentme.desktop"
DESKTOP_FILE="$DATA/applications/$DESKTOP_ID"
MIME_FILE="$DATA/mime/packages/segmentme.xml"
# hicolor is the fallback of every icon theme and every theme index lists this
# size; the PNG is larger than 256 px but GTK/Qt scale it down.
MIME_ICON="$DATA/icons/hicolor/256x256/mimetypes/application-x-segmentme-project.png"

refresh() {
    # Rebuild the caches that file managers and application menus read.
    if command -v update-mime-database >/dev/null 2>&1; then
        update-mime-database "$DATA/mime" >/dev/null
    else
        echo "  note: update-mime-database not found (package shared-mime-info);" \
             ".SEproj files may not be recognised"
    fi
    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database "$DATA/applications" >/dev/null || true
    fi
    if [ -d "$DATA/icons/hicolor" ]; then
        touch "$DATA/icons/hicolor"   # bumps the mtime so GTK re-scans the theme
    fi
}

if [ "${1:-}" = "--uninstall" ]; then
    rm -f "$DESKTOP_FILE" "$MIME_FILE" "$MIME_ICON"
    if [ -f "$CONFIG/mimeapps.list" ]; then
        sed -i "\|^$MIME_TYPE=|d" "$CONFIG/mimeapps.list"
    fi
    refresh
    echo "SegmentME desktop integration removed."
    exit 0
elif [ -n "${1:-}" ]; then
    echo "Usage: $0 [--uninstall]"
    exit 1
fi

# What the menu entry launches: the packaged executable, or main.py from a
# source checkout.
if [ -x "$HERE/SegmentME" ]; then
    EXEC="\"$HERE/SegmentME\""
    RES="$HERE/_internal/resources"
elif [ -f "$HERE/main.py" ]; then
    PYTHON="${SEGMENTME_PYTHON:-$HERE/.venv/bin/python}"
    if [ ! -x "$PYTHON" ]; then
        echo "ERROR: $PYTHON not found. Run ./install.sh first, or set SEGMENTME_PYTHON" \
             "to an interpreter that has SegmentME's dependencies."
        exit 1
    fi
    EXEC="\"$PYTHON\" \"$HERE/main.py\""
    RES="$HERE/resources"
else
    echo "ERROR: run this script from the SegmentME folder" \
         "(next to the SegmentME executable or main.py)."
    exit 1
fi

ICON_SRC="$RES/icons/desktop.png"
mkdir -p "$(dirname "$DESKTOP_FILE")" "$(dirname "$MIME_FILE")" "$(dirname "$MIME_ICON")"

# The MIME definition and the desktop entry template are shared with the
# .deb package (build.py); only the launch command and icon path differ.
cp "$RES/linux/segmentme.xml" "$MIME_FILE"
esc() { printf '%s' "$1" | sed 's/[&|\\]/\\&/g'; }   # make a value safe inside s|..|..|
sed -e "s|@EXEC@|$(esc "$EXEC")|" -e "s|@ICON@|$(esc "$ICON_SRC")|" \
    "$RES/linux/segmentme.desktop" > "$DESKTOP_FILE"
cp "$ICON_SRC" "$MIME_ICON"
refresh
if command -v xdg-mime >/dev/null 2>&1; then
    xdg-mime default "$DESKTOP_ID" "$MIME_TYPE"
else
    echo "  note: xdg-mime not found (package xdg-utils); set SegmentME as the" \
         "default for .SEproj files from your file manager's Open With dialog"
fi

echo "SegmentME registered for $USER:"
echo "  menu entry : $DESKTOP_FILE"
echo "  launches   : $EXEC"
echo "  file type  : $MIME_TYPE (*.SEproj) -> $DESKTOP_ID"
echo "Double-click a .SEproj file to open the project. Undo with: $0 --uninstall"
