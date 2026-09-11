from PyQt6.QtWidgets import QMenuBar, QDialog, QVBoxLayout, QTextEdit, QPushButton, QTextBrowser
from PyQt6.QtGui import QAction, QFont
from ui.dialogs.export_dialog import ExportDialog
from ui.dialogs.model_manager_dialog import ModelManagerDialog

from core.tools import sam_registry
from services.file_handlers import get_resource_path
from services.logger import get_logger

logger = get_logger(__name__)


class MenuBar(QMenuBar):
    """
    A custom menu bar with options for File, Actions, View, and Help menus.
    """

    def __init__(self, parent):
        """
        Initialize the MenuBar and set up the menus.

        Args:
            parent: The parent widget (typically the main window).
        """
        super().__init__(parent)
        self.parent = parent
        self.setNativeMenuBar(False)  # For cross-platform consistency

        # Initialize Menus
        self._init_file_menu()
        self._init_actions_menu()
        self._init_view_menu()
        self._init_settings_menu()
        self._init_help_menu()

    def _init_file_menu(self):
        """
        Create the File menu with options for loading, saving, exporting, and exiting.
        """
        file_menu = self.addMenu("File")

        actions = [
            ("Import Images", self.parent.import_images),
            ("Annotations", self.parent.show_results ), #self.parent.toggle_results_dock
            ("Save Results", self.parent.save_results_csv),
            ("Export annotations", self.parent._export_results), 
            ("Exit", self.parent.close),
        ]

        self._add_actions_to_menu(file_menu, actions)


    def _init_actions_menu(self):
        actions_menu = self.addMenu("Actions")

        actions = [
            ("Run Inference", lambda: self.parent.popup_inference_dialog('Running inference...')),
             ("Train Custom Model", self.parent.popup_training_dialog)
        ]

        self._add_actions_to_menu(actions_menu, actions)

    def _init_view_menu(self):
        """
        Create the View menu with options for fullscreen and other view settings.
        """
        view_menu = self.addMenu("View")

        actions = [
            ("Fullscreen Mode", None),  # Placeholder for fullscreen toggle
        ]

        self._add_actions_to_menu(view_menu, actions)

        view_menu.addSeparator()

        self.toggle_masks_action = QAction("Show Masks", self.parent)
        self.toggle_masks_action.setCheckable(True)
        self.toggle_masks_action.setChecked(True)
        self.toggle_masks_action.setToolTip("Toggle mask overlay visibility (hold H to peek)")
        self.toggle_masks_action.toggled.connect(
            lambda checked: self.parent.image_display.set_masks_visible(checked)
        )
        view_menu.addAction(self.toggle_masks_action)

    def _init_settings_menu(self):
        """
        Create the Settings menu with user-configurable preferences.
        """
        settings_menu = self.addMenu("Settings")

        actions = [
            ("Models...", self.show_models_dialog),
        ]

        self._add_actions_to_menu(settings_menu, actions)

    def show_models_dialog(self):
        """
        Manage model checkpoints and pick the model behind the SAM tool; if
        the tool is currently active, reload it with the new model right away.
        """
        dialog = ModelManagerDialog(self.parent)
        accepted = dialog.exec()
        # Models may have been downloaded even if the dialog was cancelled.
        self.parent.sidebar.update_sam_tooltip()
        if not accepted:
            return

        key = dialog.selected_key()
        if key is None or key == sam_registry.get_selected_key():
            return

        sam_registry.set_selected_key(key)
        logger.info("SAM model preference changed to %r", key)
        self.parent.sidebar.update_sam_tooltip()

        if self.parent.sidebar.sam.isChecked():
            if not self.parent.tool_manager.enable_tool("sam"):
                self.parent.sidebar.sam.setChecked(False)

    def _init_help_menu(self):
        """
        Create the Help menu with documentation and about options.
        """
        help_menu = self.addMenu("Help")

        actions = [
            ("Documentation", self.show_documentation),  # Placeholder for documentation
            ("About", None),  # Placeholder for about dialog
        ]

        self._add_actions_to_menu(help_menu, actions)

    def _add_actions_to_menu(self, menu, actions):
        """
        Utility method to add a list of actions to a given menu.

        Args:
            menu: The QMenu to which actions will be added.
            actions: A list of tuples where each tuple contains:
                - action_text (str): The text for the action.
                - callback (callable or None): The function to connect to the action.
        """
        for action_text, callback in actions:
            action = QAction(action_text, self.parent)
            if callback:
                action.triggered.connect(callback)
            menu.addAction(action)


    def show_documentation(self):
        """
        Open the user guide in a resizable, styled dialog.
        """
        doc_dialog = QDialog(self.parent)
        doc_dialog.setWindowTitle("📖 SegmentME Documentation")
        doc_dialog.resize(700, 600)  # Optimal size for readability

        # Load the formatted documentation
        with open(get_resource_path("resources/docs/user_guide.txt"), "r", encoding="utf-8") as file:
            documentation_text = file.read()

        # Create a rich-text display with styling
        text_browser = QTextBrowser(doc_dialog)
        text_browser.setReadOnly(True)
        text_browser.setHtml(documentation_text.replace("\n", "<br>"))
        text_browser.setFont(QFont("Arial", 11))

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(text_browser)

        # Close button
        close_button = QPushButton("Close", doc_dialog)
        close_button.clicked.connect(doc_dialog.close)
        layout.addWidget(close_button)

        doc_dialog.setLayout(layout)
        doc_dialog.exec()

    def show_export_dialog(self):
        dialog = ExportDialog(self.parent)
        if dialog.exec():
            settings = dialog.get_settings()

            if not settings["output_dir"]:
                logger.warning("Export aborted: no output folder selected")
                return

            #  Call your export logic
            self.parent.export_annotations(
                format=settings["format"],
                output_dir=settings["output_dir"],
                train_pct=settings["train"],
                val_pct=settings["val"],
                test_pct=settings["test"]
            )
