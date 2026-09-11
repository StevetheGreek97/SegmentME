import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PyQt6.QtGui import QPen, QColor, QPolygonF
from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsPolygonItem
from PyQt6.QtCore import pyqtSignal, QObject, QPointF
from services.logger import get_logger
from services import model_store

logger = get_logger(__name__)
from DEXTR.networks.deeplab_resnet import resnet101
from DEXTR.dataloaders.helpers import get_bbox, crop_from_bbox, fixed_resize, make_gt, cstm_normalize, crop2fullmask
class DEXTRMasker(QObject):
    """
    A class for creating masks using Deep Extreme Cut (DEXTR).
    """
    mask_added = pyqtSignal(str, np.ndarray)

    def __init__(self, parent, device: str = "cpu"):
        """
        Initialize the DEXTRMasker.

        Args:
            parent: Reference to the parent ImageDisplay instance.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        super().__init__()
        self.parent = parent
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load the DEXTR model
        self.model = self._load_dextr_model()

        # Interactive extreme points
        self.extreme_points = []
        self.point_items = []

        self.current_polygon_item = None  # Reference to displayed polygon
        self.mask = None

    def _load_dextr_model(self):
        """Load the DEXTR model once."""
        spec = model_store.MODELS["dextr"]
        model_path = model_store.find_checkpoint(spec.checkpoint)
        if model_path is None:
            raise FileNotFoundError(
                f"{spec.label} checkpoint ({spec.checkpoint}) is not installed. "
                "Download it in Settings -> Models."
            )
        logger.info("Loading DEXTR model from %s (device=%s)", model_path, self.device)

        model = resnet101(1, nInputChannels=4, classifier='psp').to(self.device)
        model.load_state_dict(torch.load(str(model_path), map_location=self.device))
        model.eval()

        return model

    def _add_graphics_point(self, point: tuple):
        """Helper function to add a point to the scene."""
        pen = QPen(QColor(0, 255, 0), 2)

        ellipse = QGraphicsEllipseItem(point[0] - 2, point[1] - 2, 4, 4)
        ellipse.setPen(pen)
        ellipse.setBrush(QColor(0, 255, 0))
        self.parent.scene.addItem(ellipse)

        return ellipse

    def add_point(self, point: tuple):
        """
        Add an extreme point for segmentation.

        Args:
            point (tuple): The point to add as (x, y).
        """
        if len(self.extreme_points) >= 4:
            logger.warning("Already have 4 extreme points; press E to segment")
            return

        ellipse = self._add_graphics_point(point)

        self.extreme_points.append(point)
        self.point_items.append(ellipse)

        logger.debug("Added extreme point %s (%d/4)", point, len(self.extreme_points))

    def display_polygon(self, polygon_points: list, color=QColor(0, 255, 0), line_width: int = 2):
        """
        Display a polygon on the scene using the provided points.

        Args:
            polygon_points (list): List of (x, y) tuples representing the polygon vertices.
            color (QColor): Color of the polygon outline.
            line_width (int): Width of the polygon lines.
        """
        if self.current_polygon_item:
            self.parent.scene.removeItem(self.current_polygon_item)
            self.current_polygon_item = None

        if not polygon_points:
            logger.warning("No polygon points to display")
            return

        polygon = QPolygonF([QPointF(x, y) for x, y in polygon_points])
        pen = QPen(color)
        pen.setWidth(line_width)

        polygon_item = QGraphicsPolygonItem(polygon)
        polygon_item.setPen(pen)
        self.parent.scene.addItem(polygon_item)

        self.current_polygon_item = polygon_item
        self.mask = np.array(polygon_points, dtype=np.float32)

    def generate_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generate a mask using the DEXTR model based on extreme points.

        Args:
            image (numpy.ndarray): The input image for mask generation.

        Returns:
            numpy.ndarray: Generated mask.
        """
        if len(self.extreme_points) != 4:
            logger.warning("DEXTR needs exactly 4 extreme points (have %d)",
                           len(self.extreme_points))
            return None

        # Convert extreme points to numpy array
        points = np.array(self.extreme_points).astype(np.int32)

        # Get bounding box
        bbox = get_bbox(image, points=points, pad=50, zero_pad=True)

        # ✅ Validate bounding box before cropping
        if bbox is None or (bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
            logger.warning("Invalid bounding box %s from extreme points; segmentation skipped", bbox)
            return None

        # Crop image and resize
        crop_image = crop_from_bbox(image, bbox, zero_pad=True)

        # ✅ Ensure cropped image is valid before proceeding
        if crop_image is None or crop_image.size == 0:
            logger.warning("Cropped image is empty; segmentation skipped")
            return None

        resize_image = fixed_resize(crop_image, (512, 512)).astype(np.float32)

        # Generate heat map
        extreme_points = points - [np.min(points[:, 0]), np.min(points[:, 1])] + [50, 50]
        extreme_points = (512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(np.int32)
        extreme_heatmap = make_gt(resize_image, extreme_points, sigma=10)
        extreme_heatmap = cstm_normalize(extreme_heatmap, 255)

        # Prepare input tensor
        input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)
        inputs = torch.from_numpy(input_dextr.transpose((2, 0, 1))[np.newaxis, ...]).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model.forward(inputs)
            outputs = F.interpolate(outputs, size=(512, 512), mode="bilinear", align_corners=True)
            outputs = outputs.cpu().numpy()[0, 0]

        # Apply threshold
        pred_mask = (1 / (1 + np.exp(-outputs)) > 0.8).astype(np.uint8)

        # ✅ Validate `pred_mask`
        if pred_mask.sum() == 0:
            logger.warning("DEXTR predicted an empty mask; nothing to save")
            return None

        # ✅ Use `crop2fullmask()` to properly restore mask instead of `crop_from_bbox()`
        full_mask = crop2fullmask(pred_mask, bbox, im_size=image.shape[:2], zero_pad=True, relax=50)

        # ✅ Ensure cropped mask is valid
        if full_mask is None or full_mask.size == 0:
            logger.warning("Restored full-size mask is empty; segmentation skipped")
            return None

        # Extract contours
        contours, _ = cv2.findContours(full_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon = contours[0].squeeze(axis=1).tolist() if contours else []

        self.display_polygon(polygon)
        self.mask = np.array(polygon, dtype=np.float32)

        return self.mask



    def clear_temp_items(self):
        """
        Clear temporary points, lines, and polygons.
        """
        for item in self.point_items:
            self.parent.scene.removeItem(item)

        self.extreme_points.clear()
        self.point_items.clear()

        if self.current_polygon_item:
            self.parent.scene.removeItem(self.current_polygon_item)
            self.current_polygon_item = None

        self.mask = None
        logger.debug("Cleared temporary items")

    def complete_mask(self):
        """
        Save the generated mask to the database.
        """
        if not self.parent.parent.sidebar.has_valid_class_selection():
            self.clear_temp_items()
            return  # Cancel saving if no valid class is selected

        if self.mask is None or self.mask.shape[0] == 0:
            logger.warning("No mask to save; press E to segment first")
            return

        if self.mask.ndim != 2 or self.mask.shape[1] != 2:
            logger.error("Invalid mask shape %s (expected Nx2); not saved", self.mask.shape)
            return

        image_name = self.parent.parent.state_manager.current_image_name
        class_name, selected_color = self.parent.parent.sidebar.get_selected_class_color()

        self.parent.parent.state_manager.mask_manager.save_mask(self.mask, image_name, class_name)

        self.mask_added.emit(image_name, self.mask)
        logger.info("Saved DEXTR mask (%d points) for image %s as class %r",
                    self.mask.shape[0], image_name, class_name)
        self.clear_temp_items()
