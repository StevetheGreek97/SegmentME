"""Intelligent Scissors: click seed points along an object's edge; the path
between them follows the image's edges (A* over an edge map at half
resolution). The mouse shows the path to the next seed. S saves."""
import heapq

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QTimer
from PyQt6.QtGui import QPolygonF, QPen, QColor
from PyQt6.QtWidgets import QGraphicsPolygonItem, QGraphicsEllipseItem

from core.tools.base import MaskTool
from services.logger import get_logger

logger = get_logger(__name__)


class IntelligentScissors(MaskTool):
    label = "scissors"

    def __init__(self, host, threshold=100):
        super().__init__(host)
        self.edge_weights = None  # of the image on screen; made on the first click
        self.seed_points = []
        self.polygon_points = []
        self.current_polygon_item = None
        self.dynamic_path_item = None
        self.path_cache = {}  # computed paths, size-limited
        self.anchor_point = None  # the last seed point
        self.scale_factor = 0.5  # the edge map is made at half resolution
        self.threshold = threshold
        self.restrictive_circle_item = None

        # Throttled drawing of the path under the mouse
        self.update_timer = QTimer()
        self.update_timer.setInterval(5)
        self.update_timer.timeout.connect(self.render_updates)
        self.pending_updates = []

    # -------------------------------------------------------------- input
    def left_click(self, point, ctrl=False):
        self.add_seed_point(point)

    def mouse_move(self, point):
        if self.seed_points:
            self.update_dynamic_path(point)

    def image_changed(self, image):
        """A new image: its edges are worked out on the first click."""
        super().image_changed(image)
        self.edge_weights = None
        self.path_cache.clear()

    def update_dynamic_path(self, cursor_position):
        """
        Update the dynamic path from the last seed or anchor point to the cursor position.
        """
        if not self.seed_points:
            return

        last_point = self.anchor_point if self.anchor_point else self.seed_points[-1]
        distance = np.hypot(cursor_position[0] - last_point[0], cursor_position[1] - last_point[1])

        if distance > self.threshold:
            # Approximate path for distant points
            dynamic_path = self.approximate_path(last_point, cursor_position)
        else:
            # Full path computation for closer points
            dynamic_path = self.compute_path(last_point, cursor_position)

        self.pending_updates = [dynamic_path]   # only the newest path matters
        if not self.update_timer.isActive():
            self.update_timer.start()

    def render_updates(self):
        """
        Render pending updates in batches to reduce rendering overhead.
        """
        if not self.pending_updates:
            self.update_timer.stop()
            return

        dynamic_path = self.pending_updates.pop(0)
        path_polygon = QPolygonF([QPointF(x, y) for x, y in dynamic_path])

        if not self.dynamic_path_item:
            pen = QPen(QColor(0, 0, 0))  # Black color
            pen.setWidth(2)
            self.dynamic_path_item = QGraphicsPolygonItem()
            self.dynamic_path_item.setPen(pen)
            self.scene.addItem(self.dynamic_path_item)

        self.dynamic_path_item.setPolygon(path_polygon)

    def set_image(self, image):
        """
        Set the current image and compute the edge map and weights.
        """
        self.path_cache.clear()
        if image is not None:
            resized_image = cv2.resize(image, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
            edge_map = cv2.Canny(gray_image, 50, 150)
            self.edge_weights = 1 / (edge_map + 1e-5)  # Avoid division by zero
            logger.debug("Computed edge map for scissors (%dx%d)", *edge_map.shape[:2])
        else:
            logger.warning("Cannot initialize scissors: no image provided")

    def add_seed_point(self, point):
        """
        Add a seed point and update the polygon.
        """
        if self.edge_weights is None:
            self.set_image(self.host.image())
            if self.edge_weights is None:
                return

        if self.seed_points:
            last_point = self.seed_points[-1]
            distance = np.hypot(point[0] - last_point[0], point[1] - last_point[1])
            if distance > self.threshold:
                logger.debug("Seed point %s rejected: outside threshold (%.0f > %d)",
                             point, distance, self.threshold)
                return

        self.seed_points.append(point)
        self.anchor_point = point  # Save this point as an anchor

        if len(self.seed_points) > 1:
            start = self.seed_points[-2]
            end = self.seed_points[-1]
            path = self.compute_path(start, end)
            self.polygon_points.extend(path)

        self.update_polygon()
        self.clear_dynamic_path()
        self.draw_restrictive_threshold(point)

    def draw_restrictive_threshold(self, center):
        """
        Draw a visual representation of the restrictive threshold around the last seed point.
        """
        if self.restrictive_circle_item:
            self.scene.removeItem(self.restrictive_circle_item)

        rect = QRectF(center[0] - self.threshold, center[1] - self.threshold, self.threshold * 2, self.threshold * 2)
        pen = QPen(QColor(255, 0, 0))  # Red color
        pen.setWidth(1)
        self.restrictive_circle_item = QGraphicsEllipseItem(rect)
        self.restrictive_circle_item.setPen(pen)
        self.scene.addItem(self.restrictive_circle_item)

    def compute_path(self, start, end):
        """
        Compute the minimum cost path between two points.
        """
        scaled_start = (int(start[0] * self.scale_factor), int(start[1] * self.scale_factor))
        scaled_end = (int(end[0] * self.scale_factor), int(end[1] * self.scale_factor))

        if (scaled_start, scaled_end) in self.path_cache:
            scaled_path = self.path_cache[(scaled_start, scaled_end)]
        else:
            scaled_path = self._astar_path(scaled_start, scaled_end)
            if len(self.path_cache) > 1000:  # Limit cache size
                self.path_cache.pop(next(iter(self.path_cache)))
            self.path_cache[(scaled_start, scaled_end)] = scaled_path

        return [(int(x / self.scale_factor), int(y / self.scale_factor)) for x, y in scaled_path]

    def approximate_path(self, start, end):
        """
        Approximate a straight-line path between two points for distant cursor movements.
        """
        x1, y1 = start
        x2, y2 = end
        num_points = int(np.hypot(x2 - x1, y2 - y1))
        line_x = np.linspace(x1, x2, num_points, dtype=int)
        line_y = np.linspace(y1, y2, num_points, dtype=int)
        return list(zip(line_x, line_y))

    def _astar_path(self, start, end):
        """
        A* pathfinding algorithm.
        """
        h, w = self.edge_weights.shape
        costs = np.full((h, w), np.inf, dtype=np.float32)
        costs[start[1], start[0]] = 0
        prev = np.full((h, w, 2), -1, dtype=np.int32)
        pq = [(0, start)]

        visited = set()  # To avoid revisiting nodes

        while pq:
            _, current = heapq.heappop(pq)
            if current == end:
                break

            if current in visited:
                continue
            visited.add(current)

            x, y = current

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                    edge_cost = self.edge_weights[ny, nx]
                    new_cost = costs[y, x] + edge_cost
                    heuristic = abs(nx - end[0]) + abs(ny - end[1])
                    total_cost = new_cost + heuristic

                    if new_cost < costs[ny, nx]:
                        costs[ny, nx] = new_cost
                        prev[ny, nx] = [x, y]
                        heapq.heappush(pq, (total_cost, (nx, ny)))

        path = []
        x, y = end
        while (x, y) != (-1, -1):
            path.append((x, y))
            x, y = prev[y, x]
        return path[::-1]

    def update_polygon(self):
        """
        Update the polygon display in the parent scene.
        """
        if self.current_polygon_item:
            self.scene.removeItem(self.current_polygon_item)
            self.current_polygon_item = None

        if not self.polygon_points:
            return

        polygon = QPolygonF([QPointF(x, y) for x, y in self.polygon_points])

        pen = QPen(QColor(0, 255, 255))  # Cyan color
        pen.setWidth(2)
        polygon_item = QGraphicsPolygonItem(polygon)
        polygon_item.setPen(pen)
        self.scene.addItem(polygon_item)
        self.current_polygon_item = polygon_item

    def clear_dynamic_path(self):
        self.pending_updates = []
        if self.dynamic_path_item:
            self.scene.removeItem(self.dynamic_path_item)
            self.dynamic_path_item = None

    def clear_polygon(self):
        self.seed_points = []
        self.polygon_points = []
        self.anchor_point = None  # Reset the anchor point

        if self.current_polygon_item:
            self.scene.removeItem(self.current_polygon_item)
            self.current_polygon_item = None

        self.clear_dynamic_path()
        logger.debug("Cleared scissors polygon")

    def remove_restrictive_threshold(self):
        """
        Remove the restrictive threshold visual indicator.
        """
        if self.restrictive_circle_item:
            self.scene.removeItem(self.restrictive_circle_item)
            self.restrictive_circle_item = None
    # ------------------------------------------------------------- saving
    def current_polygon(self):
        if not self.polygon_points:
            return None
        return np.array(self.polygon_points, dtype=np.float32)

    def clear_temp_items(self):
        self.clear_polygon()
        self.remove_restrictive_threshold()
