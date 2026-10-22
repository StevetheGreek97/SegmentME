"""The project database: named queries, and one connection per thread."""
import threading

import numpy as np
import pytest


@pytest.fixture
def state(qapp, tmp_path):
    from core.state import StateManager
    root = tmp_path / "proj"
    (root / ".segmentme").mkdir(parents=True)
    s = StateManager(db_path=str(root / ".segmentme" / "masks.db"))
    yield s
    s.db.conn.close()


def _square(x):
    return np.array([[x, 0], [x + 5, 0], [x + 5, 5], [x, 5]], np.float32)


def test_named_queries(state):
    mm = state.mask_manager
    mm.save_mask(_square(0), "a", "leaf")
    mm.save_mask(_square(10), "a", "seed")
    mm.save_mask(_square(20), "b", "leaf")
    assert mm.count() == 3
    assert sorted(mm.counts_by_class()) == [("leaf", 2), ("seed", 1)]
    assert sorted(mm.counts_by_class("a")) == [("leaf", 1), ("seed", 1)]
    assert mm.counts_per_image() == {"a": 2, "b": 1}
    assert mm.image_names_with_masks() == {"a", "b"}
    first_id = mm.load_masks("a")[0][0]
    assert mm.class_of(first_id) == "leaf" and mm.class_of(999) is None
    count, top, total = mm.fingerprint("a")
    assert count == 2 and top == max(m[0] for m in mm.load_masks("a"))
    updates = []
    state.masks_updated.connect(lambda: updates.append(1))
    mm.delete_for_images(["a"], chunk=1)
    assert mm.counts_per_image() == {"b": 1} and updates == [1]


def test_window_connection_refuses_other_threads(state):
    """Work on a worker thread must open its own connection (it could
    otherwise interleave with the window's transactions)."""
    import sqlite3
    errors = []

    def use():
        try:
            state.mask_manager.count()
        except sqlite3.ProgrammingError as exc:
            errors.append(exc)
    t = threading.Thread(target=use)
    t.start()
    t.join()
    assert errors
