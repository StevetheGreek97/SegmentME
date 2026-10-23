import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

from services.logger import get_logger

logger = get_logger(__name__)


def find_seproj(project_root: str | Path) -> Optional[Path]:
    candidates = list(Path(project_root).glob("*.SEproj"))
    return candidates[0] if candidates else None


def read_seproj(project_root: str | Path) -> Dict[str, Any]:
    path = find_seproj(project_root)
    if not path or not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8").strip()
        if text.startswith("{"):
            data = json.loads(text)
            if isinstance(data, dict):
                return data
            raise ValueError(f"expected a JSON object, found {type(data).__name__}")
        return {}  # legacy plain-text marker
    except Exception as exc:
        # The next save rewrites the file from what is known: keep the damaged
        # one for a look (and say so), instead of losing it silently.
        copy = path.with_name(f"{path.name}.damaged-{datetime.now():%Y%m%d-%H%M%S}")
        try:
            shutil.copy2(path, copy)
        except OSError:
            copy = None
        logger.error("Project file %s could not be read (%s); a copy was kept as %s",
                     path, exc, copy or "nothing (copy failed)")
        return {}


def write_seproj(project_root: str | Path, data: Dict[str, Any]):
    """Written to a temporary file first and then swapped in, so a crash or
    power cut never leaves a half-written project file."""
    path = find_seproj(project_root)
    if path is None:
        name = Path(project_root).name
        path = Path(project_root) / f"{name}.SEproj"
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(temp, path)
    except OSError:
        logger.exception("Could not write project file %s", path)
        temp.unlink(missing_ok=True)
        raise


def get_saved_images(project_root: str | Path) -> List[str]:
    return read_seproj(project_root).get("images", [])


def update_images(project_root: str | Path, image_names: List[str]):
    data = read_seproj(project_root)
    data["images"] = sorted(image_names)
    data.setdefault("name", Path(project_root).name)
    data.setdefault("created", datetime.now().isoformat())
    data["last_modified"] = datetime.now().isoformat()
    write_seproj(project_root, data)


def update_classes(project_root: str | Path, classes: List[Dict[str, str]]):
    """classes: list of {"name": "Fish", "color": "#ff0000"}"""
    data = read_seproj(project_root)
    data["classes"] = classes
    data.setdefault("name", Path(project_root).name)
    data.setdefault("created", datetime.now().isoformat())
    data["last_modified"] = datetime.now().isoformat()
    write_seproj(project_root, data)
