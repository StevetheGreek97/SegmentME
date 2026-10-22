import os
from core.database.class_manager import ClassDatabaseManager
from core.database.connection import DatabaseConnection
from core.database.mask_manager import MaskDatabaseManager
from services.logger import get_logger

from PyQt6.QtCore import QObject, pyqtSignal
from pathlib import Path

logger = get_logger(__name__)

class StateManager(QObject):
    image_changed = pyqtSignal(str)
    masks_updated = pyqtSignal()     # masks were added, removed or changed
    classes_changed = pyqtSignal()   # classes were added, removed, renumbered or recoloured
    """
    Manages the application state including the current image index, path, and associated masks and colors.
    """

    def __init__(self, db_path):
        super().__init__()  # Initialize the QObject base class
        self.project_root = Path(db_path).parent.parent

    

        # Shared Database Connection
        self.db = DatabaseConnection(db_path)

        logger.info("Opened project %s (database: %s)", self.project_root, db_path)


        # db  Managers
        self.class_manager = ClassDatabaseManager(self, self.db)
        self.mask_manager = MaskDatabaseManager(self, self.db)



        # Initialize state variables
        self.image_index = -1  # Index of the currently displayed image (-1 means no image loaded)
        self.image_paths = []
        self._current_image = None  # NumPy array of the current image

    @property
    def current_image(self):
        """
        Get the NumPy array of the current image.

        Returns:
            np.ndarray: The image data, or None if no image is loaded.
        """
        return self._current_image

    @property
    def current_image_name(self):
        """
        Get the name of the currently displayed image (without extension).

        Returns:
            str: Name of the current image, or None if no image is loaded.
        """
        if self.current_image_path:
            return os.path.splitext(os.path.basename(self.current_image_path))[0]
        return None

    @current_image.setter
    def current_image(self, image):
        """
        Set the current image.

        Args:
            image (np.ndarray): The image data.
        """
        self._current_image = image

    def set_image_paths(self, image_paths):
        """
        Set the list of image paths and reset the index.

        Args:
            image_paths (list): List of image file paths.
        """
        self.image_paths = image_paths
        self.image_index = 0 if image_paths else -1  # Reset to the first image if the list is not empty

    @property
    def current_image_path(self):
        """
        Get the path of the currently displayed image.

        Returns:
            str: Path of the current image, or None if no image is loaded.
        """
        if 0 <= self.image_index < len(self.image_paths):
            return self.image_paths[self.image_index]
        return None

    @property
    def current_image_index(self):
        """
        Get the index of the currently displayed image.

        Returns:
            int: Index of the current image, or -1 if no image is loaded.
        """
        return self.image_index

    @property
    def current_masks(self):
        """
        Get the masks associated with the currently displayed image.

        Returns:
            list: List of masks for the current image, or an empty list if no masks are available.
        """
        if not self.current_image_name:
            return []

        return self.mask_manager.load_masks(self.current_image_name)


    def next_image(self):
        """
        Navigate to the next image in the list.

        Returns:
            str: Path of the new current image, or None if at the end of the list.
        """
        if self.image_index < len(self.image_paths) - 1:
            self.image_index += 1
            self.image_changed.emit(self.current_image_path)  # Emit signal
            return self.current_image_path
        return None

    def previous_image(self):
        """
        Navigate to the previous image in the list.

        Returns:
            str: Path of the new current image, or None if at the beginning of the list.
        """
        if self.image_index > 0:
            self.image_index -= 1
            self.image_changed.emit(self.current_image_path)  # Emit signal
            return self.current_image_path
        return None

    def reset(self):
        """
        Reset all state to the initial empty values.
        """
        self.image_index = -1
        self.image_paths = []


