"""Reusable Record3D (iPhone LiDAR) camera interface.

Wraps the Record3D USB connection so the sanity check, live viewer, and dataflow
checks all share one implementation instead of duplicating the connect/grab
boilerplate. Exposes blocking (`wait`) and non-blocking (`latest`) frame access,
with RGB, metric depth, a 3x3 intrinsics matrix, and the ARKit pose.

Note: the Record3D Python library is USB-only (Wi-Fi streaming from the app does
not reach it). Enable "USB streaming" in the Record3D app and connect via cable.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from record3d import Record3DStream


@dataclass
class Record3DFrame:
    """One synchronized iPhone frame."""

    rgb: np.ndarray        # HxWx3 uint8
    depth: np.ndarray      # HxW float32, metres
    intrinsics: np.ndarray  # 3x3 K
    pose: object           # ARKit pose object (quaternion + translation)


def intrinsics_to_matrix(coeffs) -> np.ndarray:
    """Convert Record3D's IntrinsicMatrixCoeffs (fx, fy, tx, ty) to a 3x3 K.

    Record3D exposes the principal point as (tx, ty), i.e. (cx, cy).
    """
    return np.array(
        [[coeffs.fx, 0.0, coeffs.tx],
         [0.0, coeffs.fy, coeffs.ty],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


class Record3DCamera:
    """USB connection to an iPhone running Record3D (streaming mode)."""

    def __init__(self) -> None:
        self._new = threading.Event()
        self._session: Record3DStream | None = None

    @staticmethod
    def is_available() -> bool:
        """True if at least one Record3D device is connected over USB."""
        return len(Record3DStream.get_connected_devices()) > 0

    def connect(self, device_index: int = 0) -> None:
        devices = Record3DStream.get_connected_devices()
        if not devices:
            raise RuntimeError(
                "No Record3D device found. On the iPhone, open Record3D, enable "
                "USB streaming, and connect it via cable."
            )
        self._session = Record3DStream()
        self._session.on_new_frame = self._new.set
        self._session.on_stream_stopped = lambda: None
        self._session.connect(devices[device_index])

    def _read(self) -> Record3DFrame:
        s = self._session
        return Record3DFrame(
            rgb=s.get_rgb_frame(),
            depth=s.get_depth_frame(),
            intrinsics=intrinsics_to_matrix(s.get_intrinsic_mat()),
            pose=s.get_camera_pose(),
        )

    def wait(self, timeout_s: float = 5.0) -> Record3DFrame | None:
        """Block until a new frame arrives; return None on timeout."""
        if self._session is None:
            raise RuntimeError("Call connect() first.")
        if not self._new.wait(timeout=timeout_s):
            return None
        self._new.clear()
        return self._read()

    def latest(self) -> Record3DFrame | None:
        """Non-blocking: return the latest frame only if a new one is ready."""
        if self._session is None or not self._new.is_set():
            return None
        self._new.clear()
        return self._read()

    def disconnect(self) -> None:
        if self._session is not None and hasattr(self._session, "disconnect"):
            self._session.disconnect()
