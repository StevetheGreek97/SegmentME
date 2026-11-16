"""
Switching the drawing tools on and off.

SAM and DEXTR need a model loaded first: seconds for SAM2, up to a minute
for SAM3 on a CPU. That happens on a worker thread, so the window stays
usable meanwhile (it used to freeze, e.g. when a project reopened with SAM
selected). The tool's button stays pressed, the status bar says the model
is loading, and the tool starts once the model is there. Choosing another
tool (or none) meanwhile wins: the model that arrives late is dropped.
"""
import gc
import threading
import time

import torch
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import QTimer
from core.tools.manual_mask import ManualMask
from core.tools import sam_registry, sam3_loader
from services import model_store
from ui.dialogs.model_manager_dialog import ensure_model_available
from core.tools.sam2_masker import SamMasker2, load_predictor
from core.tools.sam3_masker import Sam3Masker
from core.tools.intelligent_scissors import IntelligentScissors
from services.logger import get_logger

logger = get_logger(__name__)

MODEL_TOOLS = ("sam", "dextr")


class _Loading:
    """A model being loaded on a worker thread."""

    def __init__(self, tool, spec, load):
        self.tool, self.spec = tool, spec
        self.model = None
        self.error = None
        self.done = threading.Event()
        self.started = time.perf_counter()
        # A daemon thread: quitting the app does not wait for a slow load.
        self.thread = threading.Thread(target=self._run, args=(load,), daemon=True,
                                       name=f"load-{tool}")
        self.thread.start()

    def _run(self, load):
        try:
            self.model = load()
        except BaseException as exc:          # shown to the user on the GUI thread
            self.error = exc
        finally:
            self.done.set()


class ToolManager:
    """
    Handles activation and deactivation of image processing tools.
    """
    POLL_MS = 50

    def __init__(self, parent):
        self.parent = parent
        self.current_tool = None
        self._loading = None
        self._timer = QTimer(parent)
        self._timer.setInterval(self.POLL_MS)
        self._timer.timeout.connect(self._check_loading)

    # ------------------------------------------------------------------ public
    def loading(self):
        """The tool whose model is being loaded, or None."""
        return self._loading.tool if self._loading is not None else None

    def enable_tool(self, tool) -> bool:
        """
        Enables a tool and disables others.

        Returns False when the tool could not be enabled -- its model is not
        installed and the user declined to download it, or a tool without a
        model failed to start -- so the caller can un-press the tool's button.
        For SAM and DEXTR True means "loading": the tool starts when the model
        is there (a failure then un-presses the button itself).
        """
        self.disable_tools()

        # Model-backed tools: make sure the checkpoint is on disk first
        # (offering to download it), instead of failing deep inside a loader.
        if tool == "sam":
            spec = sam_registry.get_selected_variant()
        elif tool == "dextr":
            spec = model_store.MODELS["dextr"]
        else:
            spec = None
        if spec is not None and not ensure_model_available(self.parent, spec):
            logger.info("Tool %r not enabled: %s is not installed", tool, spec.label)
            return False

        if tool in MODEL_TOOLS:
            return self._start_loading(tool, spec)

        error = None
        try:
            host = self.parent.image_display.tool_host
            if tool == "manual_mask":
                self.current_tool = ManualMask(host)
            elif tool == "intelligent_scissors":
                self.current_tool = IntelligentScissors(host)   # edges made on the first click
            else:
                logger.warning("Unknown tool requested: %r", tool)
            if self.current_tool:
                logger.info("Enabled tool: %s", tool)
        except Exception as exc:
            logger.exception("Failed to enable tool %r", tool)
            self.current_tool = None
            error = exc
        if error is not None:
            self._tell_failed(tool, error)
            return False
        return self.current_tool is not None

    def disable_tools(self):
        """
        Disables all active tools and frees the memory held by AI models.
        A model still loading is dropped when it arrives.
        """
        if self._loading is not None:
            logger.info("Loading the %s model is no longer needed", self._loading.tool)
            self._loading = None
            self._timer.stop()
            self._show("", 0)
        if self.current_tool is None:
            return
        self.current_tool.clear_temp_items()
        was_sam = isinstance(self.current_tool, SamMasker2)
        self.current_tool = None
        if was_sam:
            self._free_sam_memory()

    # ------------------------------------------------------------ model tools
    def _start_loading(self, tool, spec):
        if tool == "sam":
            if spec.family == "sam3":
                load = lambda: sam3_loader.load_sam3_predictor(torch.device("cpu"))  # noqa: E731
            else:
                load = lambda: load_predictor("cpu", spec)  # noqa: E731
            name = spec.label
        else:
            def load():
                # Imported here: the DEXTR package is installed separately
                # (build.py); without it only this tool is unavailable.
                from core.tools.dextr_mask import load_dextr_model
                return load_dextr_model("cpu")
            name = "DEXTR"
        logger.info("Loading the %s model for tool %s on a worker thread", name, tool)
        self._loading = _Loading(tool, spec, load)
        self._show(f"Loading {name}… (the window stays usable; the tool starts when it is ready)", 0)
        self._timer.start()
        return True

    def _check_loading(self):
        loading = self._loading
        if loading is None:
            self._timer.stop()
            return
        if not loading.done.is_set():
            return
        self._timer.stop()
        self._loading = None
        seconds = time.perf_counter() - loading.started
        tool = loading.tool
        error = loading.error
        if error is None:
            try:
                host = self.parent.image_display.tool_host
                if tool == "sam":
                    logger.info("SAM tool using variant %r", loading.spec.key)
                    if loading.spec.family == "sam3":
                        self.current_tool = Sam3Masker(host, predictor=loading.model)
                    else:
                        self.current_tool = SamMasker2(host, variant=loading.spec,
                                                       predictor=loading.model)
                else:
                    from core.tools.dextr_mask import DEXTRMasker
                    self.current_tool = DEXTRMasker(host, model=loading.model)
            except Exception as exc:
                logger.exception("Failed to start tool %r", tool)
                self.current_tool = None
                error = exc
        if error is not None:
            if not isinstance(error, Exception):
                logger.error("Loading the %s model stopped: %r", tool, error)
            else:
                logger.error("Failed to enable tool %r (model load error?): %s", tool, error,
                             exc_info=(type(error), error, error.__traceback__))
            self._show("", 0)
            self._uncheck(tool)
            self._tell_failed(tool, error)
            return
        logger.info("Enabled tool: %s (model loaded in %.1f s)", tool, seconds)
        self._show(f"{loading.spec.label if tool == 'sam' else 'DEXTR'} is ready", 3000)

    # ------------------------------------------------------------------ helpers
    def _show(self, text, ms):
        status = getattr(self.parent, "statusBar", None)
        if status is not None:
            if text:
                status().showMessage(text, ms)
            else:
                status().clearMessage()

    def _uncheck(self, tool):
        button = getattr(getattr(self.parent, "sidebar", None), "_tool_buttons", {}).get(tool)
        if button is not None:
            button.setChecked(False)

    def _tell_failed(self, tool, error):
        QMessageBox.critical(
            self.parent, "Could not start tool",
            f"The {tool} tool could not be started:\n\n{error}")

    @staticmethod
    def _free_sam_memory():
        sam3_loader.unload_sam3_predictor()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("SAM model offloaded from memory")
