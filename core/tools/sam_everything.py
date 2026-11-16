"""
"Everything in a box": every object SAM finds inside a region (F key with
the SAM tool, see core/tools/sam2_masker.py; F needs a box). find_objects
also accepts no box, meaning the whole image.

How it works:
  1. SAM looks at a fixed-size picture (1024 px for SAM2). A region that
     fits (with a margin of context) is cut out and given to SAM once. A
     bigger region -- or the whole image -- is shrunk to fit (F: one look,
     a few seconds; specks can be missed), but only as far as the objects
     stay big enough for SAM: their size comes from the masks already
     saved in the project. A big scan of small animals is therefore split
     into overlapping pieces, each shrunk only that far. SAM takes about
     1.5 s per piece on a CPU to take it in, the slow part.
  2. In each piece SAM is asked at a grid of points, one point at a time.
     A point that lands on an object already found, on the background, or
     on something a red point removed is skipped: same answer again.
  3. Each answer is kept only if SAM is sure of it (its own quality score
     and a stable outline), it is not tiny, not most of the piece (that is
     the background), and its centre lies inside the piece's own part of
     the region (so each object belongs to one piece). In a split region an
     object cut by the piece's edge is left to the neighbouring piece.
  4. Overlapping answers (the same object twice, or a part of an object
     found as well as the whole) are reduced to the best one.

The user's own points steer it:
  * green (left click) points are asked first and always kept, so an object
    the grid misses can be added;
  * red (right click) points remove every object that covers them (the
    dish, the label, background).
Objects already saved on the image are not found again.

Works with any predictor that has SAM2ImagePredictor's set_image / predict
(SAM2, and SAM3 through its adapter). Touches no Qt objects: it runs on a
worker thread.
"""
import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np

from services.logger import get_logger

logger = get_logger(__name__)

MAX_SIDE = 1024          # SAM's own input size: bigger regions are split into pieces
MARGIN = 0.1             # context around a small region, as part of its size
MIN_MARGIN_PX = 16
DEFAULT_MIN_SCALE = 0.4  # shrink at most this far when the object size is unknown
TARGET_OBJECT_PX = 16    # small objects (typical sqrt(area)) should stay this big for SAM
MIN_SCALE_LIMIT = 0.1
PIECE_MARGIN = 160       # context around each piece of a split region: objects up
                         # to about twice this size are never cut
GRID_STEP = 16           # pixels between grid points
MIN_GRID, MAX_GRID = 4, 64
MIN_SCORE = 0.70         # SAM's own guess of how good a mask is
MIN_STABILITY = 0.85     # outline barely moves when the threshold moves
MIN_AREA_PX = 25         # smaller than this is a speck
MAX_REGION_SHARE = 0.6   # bigger than this part of the piece is background
MAX_OVERLAP = 0.4        # of the smaller mask: the same object (or a part of it)
SKIP_INSIDE_SHARE = 0.1  # grid points inside an object up to this size are skipped
MIN_REGION_PX = 16


class Cancelled(Exception):
    """The search was stopped (another image, another tool, Esc)."""


@dataclass
class Candidate:
    x0: int
    y0: int
    mask: np.ndarray      # bool, the bounding box of the object only
    score: float
    forced: bool          # from a green point: always kept
    area: int
    rest: tuple = (0.0, 0.0)   # what rounding x0, y0 into the common frame left over

    @property
    def x1(self):
        return self.x0 + self.mask.shape[1]

    @property
    def y1(self):
        return self.y0 + self.mask.shape[0]

    def contains(self, x, y) -> bool:
        xi, yi = int(x) - self.x0, int(y) - self.y0
        return 0 <= yi < self.mask.shape[0] and 0 <= xi < self.mask.shape[1] and bool(self.mask[yi, xi])

    def overlap(self, other) -> int:
        """Pixels the two masks share."""
        x0, y0 = max(self.x0, other.x0), max(self.y0, other.y0)
        x1, y1 = min(self.x1, other.x1), min(self.y1, other.y1)
        if x0 >= x1 or y0 >= y1:
            return 0
        a = self.mask[y0 - self.y0:y1 - self.y0, x0 - self.x0:x1 - self.x0]
        b = other.mask[y0 - other.y0:y1 - other.y0, x0 - other.x0:x1 - other.x0]
        return int(np.count_nonzero(a & b))


@dataclass
class Piece:
    core: tuple           # the part of the region this piece answers for
    window: tuple         # what SAM is shown: the core plus context
    split: bool           # one of several pieces (objects cut by its edge are left out)
    scale: float = 1.0    # below 1: the window is shrunk to what SAM sees (quick look)


@dataclass
class _Stats:
    asked: int = 0
    skipped: int = 0
    rejected: int = 0
    cut: int = 0
    embed_s: List[float] = field(default_factory=list)


def normalize_box(box, shape):
    """(x0, y0, x1, y1) as ints inside an image of `shape`, corners in order
    (None: the whole image); None when the box is too small to search."""
    H, W = shape[:2]
    if box is None:
        box = (0, 0, W, H)
    x1, y1, x2, y2 = (float(v) for v in box)
    x0, x1 = sorted((x1, x2))
    y0, y1 = sorted((y1, y2))
    x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1, y1 = min(W, int(np.ceil(x1))), min(H, int(np.ceil(y1)))
    if x1 - x0 < MIN_REGION_PX or y1 - y0 < MIN_REGION_PX:
        return None
    return x0, y0, x1, y1


def context_window(region, shape):
    """A small region plus a margin of context, inside the image."""
    H, W = shape[:2]
    x0, y0, x1, y1 = region
    mx = max(MIN_MARGIN_PX, int((x1 - x0) * MARGIN))
    my = max(MIN_MARGIN_PX, int((y1 - y0) * MARGIN))
    return max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my)


def min_scale_for(object_sizes) -> Optional[float]:
    """How far an image may be shrunk so that the smaller of the typical
    objects (`object_sizes`: sqrt of their areas, in image pixels) stay big
    enough for SAM; None without sizes."""
    sizes = [float(v) for v in object_sizes if v and v > 0]
    if len(sizes) < 3:
        return None
    small = float(np.percentile(sizes, 10))
    return float(np.clip(TARGET_OBJECT_PX / max(small, 1.0), MIN_SCALE_LIMIT, 1.0))


def plan_pieces(region, shape, min_scale=None) -> List[Piece]:
    """How SAM looks at the region. One piece when it (with context) fits
    what SAM sees. A bigger region is shrunk to fit -- one quick look -- if
    that keeps it at `min_scale` or more (DEFAULT_MIN_SCALE when unknown);
    otherwise it is split into overlapping pieces, each shrunk to
    `min_scale` (1.0: pieces at full resolution). All pieces of one search
    have the same scale."""
    window = context_window(region, shape)
    side = max(window[2] - window[0], window[3] - window[1])
    if side <= MAX_SIDE:
        return [Piece(region, window, False)]
    limit = min_scale or DEFAULT_MIN_SCALE
    if MAX_SIDE / side >= limit:
        return [Piece(region, window, False, MAX_SIDE / side)]
    H, W = shape[:2]
    x0, y0, x1, y1 = region
    margin = int(round(PIECE_MARGIN / limit))                 # image pixels
    core = int(MAX_SIDE / limit) - 2 * margin
    xs = np.linspace(x0, x1, math.ceil((x1 - x0) / core) + 1).round().astype(int)
    ys = np.linspace(y0, y1, math.ceil((y1 - y0) / core) + 1).round().astype(int)
    pieces = []
    for cy0, cy1 in zip(ys[:-1], ys[1:]):
        for cx0, cx1 in zip(xs[:-1], xs[1:]):
            c = (int(cx0), int(cy0), int(cx1), int(cy1))
            w = (max(0, c[0] - margin), max(0, c[1] - margin),
                 min(W, c[2] + margin), min(H, c[3] + margin))
            pieces.append(Piece(c, w, True, limit))
    return pieces


def grid_points(region, scale=1.0):
    """Cell centres of a grid over the region, about GRID_STEP apart."""
    x0, y0, x1, y1 = region
    nx = int(np.clip(round((x1 - x0) * scale / GRID_STEP), MIN_GRID, MAX_GRID))
    ny = int(np.clip(round((y1 - y0) * scale / GRID_STEP), MIN_GRID, MAX_GRID))
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    return [(float(x), float(y)) for y in ys for x in xs]


def _stability(low_res_logits) -> float:
    """How much of the mask survives a stricter threshold (SAM's stability score)."""
    if low_res_logits is None:
        return 1.0
    loose = np.count_nonzero(low_res_logits > -1.0)
    return float(np.count_nonzero(low_res_logits > 1.0)) / loose if loose else 0.0


def _best_mask(masks, scores, low_res):
    """The answer SAM scores best (it gives three for a single point)."""
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None]
    if scores is None:
        scores = np.ones(len(masks), np.float32)
    scores = np.asarray(scores, np.float32).reshape(-1)
    k = int(np.argmax(scores))
    logits = None
    if low_res is not None:
        low_res = np.asarray(low_res)
        logits = low_res[k] if low_res.ndim == 3 else low_res
    return masks[k] > 0, float(scores[k]), logits


def _to_candidate(mask, score, forced):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = mask[y0:y1, x0:x1].copy()
    return Candidate(int(x0), int(y0), crop, score, forced, int(len(xs)))


def _polygon(candidate, offset=(0, 0), scale=1.0):
    """Outline of the largest piece, in image pixels (the candidate is in
    pixels shrunk by `scale` from `offset`)."""
    contours, _ = cv2.findContours(candidate.mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 3:
        return None
    xy = largest.squeeze(axis=1).astype(np.float64) + (candidate.x0 + candidate.rest[0],
                                                       candidate.y0 + candidate.rest[1])
    return (xy / scale + np.asarray(offset, np.float64)).astype(np.float32)


def _rasterize(polygons, offset, shape):
    """Saved masks drawn into a window starting at `offset`."""
    canvas = np.zeros(shape, np.uint8)
    for poly in polygons:
        poly = np.asarray(poly, np.float32).reshape(-1, 2)
        if len(poly) < 3:
            continue
        cv2.fillPoly(canvas, [np.round(poly - offset).astype(np.int32)], 1)
    return canvas.astype(bool)


def _within(points, rect):
    x0, y0, x1, y1 = rect
    return [(x, y) for x, y in points if x0 <= x < x1 and y0 <= y < y1]


def _search_piece(predictor, image, piece, positives, negatives, existing, tick, cancelled,
                  stats) -> List[Candidate]:
    """The objects of one piece (before the overlaps are sorted out): in
    image pixels for a full-resolution piece, in the piece's own shrunk
    pixels for a quick look (see find_objects)."""
    wx0, wy0, wx1, wy1 = piece.window
    s = piece.scale
    crop = np.ascontiguousarray(image[wy0:wy1, wx0:wx1])
    if s < 1.0:
        size = (max(1, min(MAX_SIDE, round((wx1 - wx0) * s))),
                max(1, min(MAX_SIDE, round((wy1 - wy0) * s))))
        crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    ch, cw = crop.shape[:2]
    H, W = image.shape[:2]

    def local(points):
        return [((x - wx0) * s, (y - wy0) * s) for x, y in points]

    (rx0, ry0), (rx1, ry1) = local([piece.core[:2], piece.core[2:]])
    region_area = (rx1 - rx0) * (ry1 - ry0)
    min_area = MIN_AREA_PX * max(s, 0.25) ** 2
    inside = lambda x, y: rx0 <= x < rx1 and ry0 <= y < ry1  # noqa: E731
    pos = local(_within(positives, piece.core))      # each green point belongs to one piece
    neg = local(_within(negatives, piece.window))
    taken = (_rasterize([np.asarray(p, np.float32).reshape(-1, 2) * s for p in existing],
                        np.array([wx0, wy0], np.float32) * s, (ch, cw))
             if len(existing) else None)
    # Edges of the window that are not edges of the image: an object touching
    # one continues outside what SAM saw.
    open_edges = (wx0 > 0, wy0 > 0, wx1 < W, wy1 < H)

    t0 = time.perf_counter()
    predictor.set_image(crop)
    stats.embed_s.append(time.perf_counter() - t0)

    candidates: List[Candidate] = []
    background: List[Candidate] = []     # too big to be an object, unwanted, or cut
    for (x, y), forced in [(p, True) for p in pos] + [(p, False) for p in
                                                      grid_points((rx0, ry0, rx1, ry1))]:
        if cancelled is not None and cancelled():
            raise Cancelled()
        tick()
        if not forced:
            if taken is not None and taken[min(int(y), ch - 1), min(int(x), cw - 1)]:
                stats.skipped += 1
                continue
            if any(c.area <= SKIP_INSIDE_SHARE * region_area and c.contains(x, y) for c in candidates):
                stats.skipped += 1
                continue
            if any(b.contains(x, y) for b in background):
                stats.skipped += 1             # the background (or a cut object) again
                continue
        stats.asked += 1
        masks, scores, low_res = predictor.predict(
            point_coords=np.array([[x, y]], np.float32), point_labels=np.array([1], np.int32),
            multimask_output=True)
        mask, score, logits = _best_mask(masks, scores, low_res)
        cand = _to_candidate(mask, score, forced)
        if cand is None or cand.area < min_area:
            stats.rejected += 1
            continue
        # Unwanted (under a red point) or the background: points inside it
        # would get the same answer, so they are skipped from now on.
        if any(cand.contains(nx, ny) for nx, ny in neg):
            background.append(cand)
            stats.rejected += 1
            continue
        if not forced:
            if cand.area > MAX_REGION_SHARE * region_area:
                background.append(cand)
                stats.rejected += 1
                continue
            if piece.split and ((open_edges[0] and cand.x0 == 0) or (open_edges[1] and cand.y0 == 0)
                                or (open_edges[2] and cand.x1 == cw)
                                or (open_edges[3] and cand.y1 == ch)):
                background.append(cand)        # cut by the edge: a neighbour sees it whole
                stats.cut += 1
                continue
            cy, cx = cand.y0 + cand.mask.shape[0] / 2, cand.x0 + cand.mask.shape[1] / 2
            if (score < MIN_SCORE or _stability(logits) < MIN_STABILITY
                    or not inside(cx, cy) or not cand.contains(x, y)):
                stats.rejected += 1
                continue
            if taken is not None:
                region_taken = taken[cand.y0:cand.y1, cand.x0:cand.x1]
                if np.count_nonzero(region_taken & cand.mask) > MAX_OVERLAP * cand.area:
                    stats.rejected += 1
                    continue
        candidates.append(cand)
    # Into the search's common frame: image pixels times the scale (the
    # same for every piece), so pieces can be compared with each other.
    dx, dy = int(round(wx0 * s)), int(round(wy0 * s))
    for cand in candidates:
        cand.x0 += dx
        cand.y0 += dy
        cand.rest = (wx0 * s - dx, wy0 * s - dy)
    return candidates


def _grid_size(piece):
    """Number of grid points of a piece (in what SAM sees)."""
    x0, y0, x1, y1 = piece.core
    s = piece.scale
    return len(grid_points((0, 0, (x1 - x0) * s, (y1 - y0) * s)))


def find_objects(predictor, image, box=None, positives: Sequence = (), negatives: Sequence = (),
                 existing: Sequence = (), progress: Optional[Callable[[int, int], None]] = None,
                 cancelled: Optional[Callable[[], bool]] = None,
                 min_scale: Optional[float] = None) -> List[np.ndarray]:
    """Every object inside `box` (x0, y0, x1, y1 in image pixels; None: the
    whole image) as a list of outlines (N x 2 float32, image pixels), top
    to bottom.

    A region bigger than what SAM sees is shrunk to fit (one look: quick),
    but never below `min_scale` (see min_scale_for: objects must stay big
    enough for SAM); past that it is searched piece by piece, each piece
    shrunk to `min_scale`.

    positives / negatives: the user's green / red points (image pixels).
    existing: outlines already saved on the image (not found again).
    progress(done, total) is called as points are tried; cancelled() is
    checked between points (raises Cancelled).
    """
    started = time.perf_counter()
    region = normalize_box(box, image.shape)
    if region is None:
        logger.info("Find all: box too small (%s)", box)
        return []
    pieces = plan_pieces(region, image.shape, min_scale)
    positives = _within(positives, region)
    total = len(positives) + sum(_grid_size(p) for p in pieces)
    done = [0]

    def tick():
        if progress is not None:
            progress(done[0], total)
        done[0] += 1

    stats = _Stats()
    candidates: List[Candidate] = []
    for piece in pieces:
        candidates += _search_piece(predictor, image, piece, positives, negatives, existing,
                                    tick, cancelled, stats)
    if progress is not None:
        progress(total, total)

    # Overlaps: green points first, then the masks SAM is surest of.
    candidates.sort(key=lambda c: (not c.forced, -c.score))
    kept: List[Candidate] = []
    for cand in candidates:
        if any(k.overlap(cand) > MAX_OVERLAP * min(k.area, cand.area) for k in kept):
            continue
        kept.append(cand)

    scale = pieces[0].scale
    polygons = [p for p in (_polygon(c, (0, 0), scale) for c in kept) if p is not None]
    polygons.sort(key=lambda p: (float(p[:, 1].min()), float(p[:, 0].min())))
    logger.info("Find all: %d object(s) in region %s, %s, %.1f s (SAM looked at the "
                "pieces in %.1f s; %d of %d points asked, %d skipped, %d answers rejected, "
                "%d cut by a piece edge, %d duplicates)", len(polygons), region,
                f"{len(pieces)} piece(s) at scale {scale:.2f}",
                time.perf_counter() - started, sum(stats.embed_s), stats.asked, total,
                stats.skipped, stats.rejected, stats.cut, len(candidates) - len(kept))
    return polygons
