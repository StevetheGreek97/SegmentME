"""
The app as the drawing tools see it (see core/tools/base.py): the image
view's scene, the image on screen, the class chosen in the sidebar, and the
project database.
"""


class ViewToolHost:
    def __init__(self, view):
        self._view = view

    @property
    def _window(self):
        return self._view.parent

    @property
    def scene(self):
        return self._view.scene

    def image(self):
        return self._window.state_manager.current_image

    def image_name(self):
        return self._window.state_manager.current_image_name

    def choose_class(self):
        """The class selected in the sidebar; None after telling the user
        there is none (no classes yet / none selected)."""
        sidebar = self._window.sidebar
        if not sidebar.has_valid_class_selection():
            return None
        class_name, _color = sidebar.get_selected_class_color()
        return class_name

    def save_mask(self, polygon, image_name, class_name):
        self._window.state_manager.mask_manager.save_mask(polygon, image_name, class_name)
