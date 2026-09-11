import gc

import torch
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt
from core.tools.manual_mask import ManualMask
from core.tools import sam_registry, sam3_loader
from services import model_store
from ui.dialogs.model_manager_dialog import ensure_model_available
from core.tools.sam2_masker import SamMasker2
from core.tools.sam3_masker import Sam3Masker
from core.tools.intellignent_scissors import IntelligentScissors
from core.tools.dextr_mask import DEXTRMasker
from services.logger import get_logger

logger = get_logger(__name__)

class ToolManager:
    """
    Handles activation and deactivation of image processing tools.
    """

    def __init__(self, parent):
        self.parent = parent
        self.current_tool = None

    def enable_tool(self, tool) -> bool:
        """
        Enables a tool and disables others.

        Returns False when the tool could not be enabled -- its model is not
        installed and the user declined to download it, or the model failed
        to load -- so the caller can un-press the tool's button.
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

        # Show wait cursor
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        error = None
        try:
            if tool == "manual_mask":
                self.current_tool = ManualMask(self.parent.image_display)
            elif tool == "sam":
                variant = spec
                logger.info("SAM tool using variant %r", variant.key)
                if variant.family == "sam3":
                    self.current_tool = Sam3Masker(self.parent.image_display)
                else:
                    self.current_tool = SamMasker2(self.parent.image_display,
                                                   variant=variant)
            elif tool == "dextr":
                self.current_tool = DEXTRMasker(self.parent.image_display)
            elif tool == "intelligent_scissors":
                self.current_tool = IntelligentScissors(self.parent.image_display)
                self.current_tool.set_image(self.parent.state_manager.current_image)
            else:
                logger.warning("Unknown tool requested: %r", tool)

            if self.current_tool:
                self.current_tool.mask_added.connect(self.parent.image_display.refresh_overlay)
                logger.info("Enabled tool: %s", tool)
        except Exception as exc:
            logger.exception("Failed to enable tool %r (model load error?)", tool)
            self.current_tool = None
            error = exc
        finally:
            QApplication.restoreOverrideCursor()

        if error is not None:
            QMessageBox.critical(
                self.parent, "Could not start tool",
                f"The {tool} tool could not be started:\n\n{error}")
            return False
        return self.current_tool is not None

    def disable_tools(self):
        """
        Disables all active tools and frees the memory held by AI models.
        """
        if self.current_tool is None:
            return
        self.current_tool.clear_temp_items()
        was_sam = isinstance(self.current_tool, SamMasker2)
        self.current_tool = None
        if was_sam:
            sam3_loader.unload_sam3_predictor()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("SAM model offloaded from memory")
