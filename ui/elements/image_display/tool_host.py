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

    def save_masks(self, polygons, image_name, class_name):
        """Several masks at once (one transaction, one redraw)."""
        return self._window.state_manager.mask_manager.save_masks(polygons, image_name, class_name)

    def typical_object_sizes(self, limit=2000):
        """Sizes (sqrt of the area, image pixels) of the most recent masks in
        the project: how big the objects are, for SAM's find-all (F)."""
        rows = self._window.state_manager.db.fetch_all(
            "SELECT surface_area FROM masks WHERE surface_area > 0 ORDER BY id DESC LIMIT ?",
            (int(limit),))
        return [float(area) ** 0.5 for (area,) in rows]

    def existing_polygons(self):
        """Outlines already saved on the image on screen."""
        name = self.image_name()
        if not name:
            return []
        return [mask for _id, mask, _cls, _area in
                self._window.state_manager.mask_manager.load_masks(name)]
