"""Registry of the SAM model variants the unified SAM tool can run.

One sidebar button ("SAM") serves every variant; which one it loads is a
user preference persisted with QSettings and edited via
Settings -> Models in the menu bar.

The variants themselves are the SAM entries of services.model_store, which
also knows where checkpoints live and how to download them. A variant is
only usable once its checkpoint is installed -- `is_available` checks that,
and ToolManager offers to download a missing one when the tool is enabled.
"""
from services import model_store
from services.model_store import ModelSpec as SamVariant  # noqa: F401  (kept for callers)

_SETTINGS_KEY = "sam/model"
_SETTINGS_ORG = "SegmentME"
_LEGACY_SETTINGS_ORG = "AquaVision"  # the app's name before the rename

SAM_VARIANTS = {
    key: spec for key, spec in model_store.MODELS.items()
    if spec.family in ("sam2", "sam3")
}

DEFAULT_KEY = "sam2_tiny"


def checkpoint_path(variant):
    """Where the variant's checkpoint is (or would be once downloaded)."""
    return model_store.checkpoint_path(variant)


def is_available(variant) -> bool:
    """True if the variant's checkpoint file is installed."""
    return model_store.is_available(variant)


def get_selected_key() -> str:
    """The persisted user choice, or the default if nothing valid is saved.

    The chosen checkpoint may not be installed yet: the SAM tool offers to
    download it when enabled, so a user who picked SAM2 Large keeps that
    choice instead of being silently bounced back to Tiny.
    """
    from PyQt6.QtCore import QSettings  # lazy: keeps worker subprocesses free of PyQt6

    key = QSettings(_SETTINGS_ORG, _SETTINGS_ORG).value(_SETTINGS_KEY)
    if key is None:
        # Preference saved while the app was still called AquaVision.
        key = QSettings(_LEGACY_SETTINGS_ORG, _LEGACY_SETTINGS_ORG).value(_SETTINGS_KEY, DEFAULT_KEY)
    return key if key in SAM_VARIANTS else DEFAULT_KEY


def set_selected_key(key: str):
    from PyQt6.QtCore import QSettings

    if key not in SAM_VARIANTS:
        raise ValueError(f"Unknown SAM variant {key!r}")
    QSettings(_SETTINGS_ORG, _SETTINGS_ORG).setValue(_SETTINGS_KEY, key)


def get_selected_variant():
    return SAM_VARIANTS[get_selected_key()]
