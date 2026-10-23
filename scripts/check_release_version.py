"""Stop a release whose tag does not match version.py.

v2.3.0 was tagged while version.py still said 2.2.1, so every file of that
release is named 2.2.1 and the app inside calls itself 2.2.1 -- which also
confuses the update check. The Build workflow runs this first:

    python scripts/check_release_version.py "$GITHUB_REF_NAME"

Exit code 0 when the tag (e.g. "v2.3.1") matches version.py, 1 otherwise.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from version import __version__  # noqa: E402


def check(tag, version=__version__):
    """None when `tag` names `version`, else the message to show."""
    if tag.lstrip("v") == version:
        return None
    return (f"The tag {tag} does not match version.py ({version}). Set __version__ = "
            f'"{tag.lstrip("v")}" in version.py, commit, and tag again '
            f"(delete the wrong tag with: git push --delete origin {tag}; git tag -d {tag}).")


def main(argv):
    if len(argv) != 2:
        print("usage: check_release_version.py <tag>")
        return 2
    problem = check(argv[1])
    if problem:
        print(f"::error title=Version mismatch::{problem}")
        return 1
    print(f"Tag {argv[1]} matches version.py ({__version__}).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
