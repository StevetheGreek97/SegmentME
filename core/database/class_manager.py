from PyQt6.QtGui import QColor

from services.logger import get_logger

logger = get_logger(__name__)


class ClassDatabaseManager:
    """
    Manages class storage and retrieval.
    """

    def __init__(self, parent,db_connection):
        """
        Initialize with a shared database connection.
        """
        self.db = db_connection
        self.parent = parent  # the StateManager: its classes_changed signal is emitted here

    def add_class(self, class_name, color):
        """
        Add a new class with a unique name.

        Args:
            class_name (str): The class name.
            color (QColor): The class color (stored as HEX string).
        """
        self.db.execute_query(
            "INSERT OR IGNORE INTO classes (class_name, color) VALUES (?, ?)",
            (class_name, color.name())
        )
        self.parent.classes_changed.emit()

    def remove_class(self, class_name):
        """
        Remove a class by name.
        """
        self.db.execute_query("DELETE FROM classes WHERE class_name = ?", (class_name,))
        self.parent.classes_changed.emit()

    def get_class_id(self, class_name):
        """
        Get the ID of a class by name.

        Returns:
            int: The class ID or None if not found.
        """
        result = self.db.fetch_one("SELECT id FROM classes WHERE class_name = ?", (class_name,))
        return result[0] if result else None

    def set_class_color(self, class_name, color):
        """Give a class a new colour (QColor or "#rrggbb"). Everything that
        shows class colours follows classes_changed."""
        hex_color = QColor(color).name()
        self.db.execute_query("UPDATE classes SET color = ? WHERE class_name = ?",
                              (hex_color, class_name))
        logger.info("Class %r is now %s", class_name, hex_color)
        self.parent.classes_changed.emit()

    def get_class_color(self, class_name):
        """
        Get the color of a class.

        Returns:
            QColor: The class color or white if not found.
        """
        result = self.db.fetch_one("SELECT color FROM classes WHERE class_name = ?", (class_name,))
        return QColor(result[0]) if result else QColor(255, 255, 255)

    def list_classes(self):
        """
        Get all class names and colors.

        Returns:
            list: A list of (id, class_name, color).
        """
        return self.db.fetch_all("SELECT * FROM classes")
    
    def get_all_class_names(self):
        """
        Retrieve all class names from the database.

        Returns:
            list: A list of class names.
        """
        return [row[1] for row in self.db.fetch_all("SELECT * FROM classes")]

    def get_idx_by_name(self, class_name):
        """
        Get the class index (ID) by class name.

        Args:
            class_name (str): The class name.

        Returns:
            int: The class ID or -1 if not found.
        """
        result = self.db.fetch_one("SELECT id FROM classes WHERE class_name = ?", (class_name,))
        return result[0] if result else -1  #  Return -1 instead of None for safety
    
    def reindex_classes(self):
        """
        Reindex mask IDs to remove gaps after deletions.
        This resets IDs sequentially (1, 2, 3, ...) and updates SQLite's internal counter.
        """
        self.db.execute_query("PRAGMA foreign_keys = OFF;")  # Disable foreign keys to prevent errors

        self.db.execute_query("""
            WITH Renumbered AS (
                SELECT id, 
                    ROW_NUMBER() OVER (ORDER BY id) AS new_id
                FROM classes
            )
            UPDATE classes
            SET id = (SELECT new_id FROM Renumbered WHERE Renumbered.id = classes.id);
        """)

        self.db.execute_query("DELETE FROM sqlite_sequence WHERE name = 'classes';")  # Reset SQLite's internal counter

        self.db.execute_query("PRAGMA foreign_keys = ON;")  # Re-enable foreign key constraints

        logger.debug("Reindexed class ids")