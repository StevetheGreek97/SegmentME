"""
Where the user left off in a project: the image on screen, the tool and
the class. Kept in <project>/.segmentme/last_place.json, so it travels with
the project; reopening it returns there (see MainApp.remember_place and
recent_projects.initialize_project).
"""
import json
import os
from pathlib import Path

from services.logger import get_logger

logger = get_logger(__name__)

FILE_NAME = "last_place.json"


def _path(project_root) -> Path:
    return Path(project_root) / ".segmentme" / FILE_NAME


def load(project_root) -> dict:
    """{"image": "scan_07.png", "tool": "sam" | None, "class": "Daphnia"};
    {} when there is nothing (yet) or the file can't be read."""
    path = _path(project_root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("Could not read %s (%s); starting at the first image", path, exc)
        return {}


def save(project_root, place: dict) -> bool:
    path = _path(project_root)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(json.dumps(place, indent=2), encoding="utf-8")
        os.replace(temp, path)
        return True
    except OSError as exc:              # read-only drive etc.: not worth bothering the user
        logger.debug("Could not save %s: %s", path, exc)
        return False
