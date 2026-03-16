# motion_profile.py
# Builds Motion Profile (MP) and Vehicle Width Profile (WP) from raw video.
# Based on Fig. 3 and Section II-B of Lin & Zheng (2023).
#
# MP(x,t): horizontal belt at the horizon height, accumulated over frames
#           vehicles appear as diagonal streaks (their trajectories)
# WP(x,t): vehicle bounding-box widths painted into the same space
#           width encodes approximate depth (wider = closer)
#
# Combined 4-channel input for ST2CN: [MP_R, MP_G, MP_B, WP]
#
# Note: this module is for inference on new video. Training uses pre-built
# .npy arrays from prepare_dataset.py, not this module directly.

import cv2
import numpy as np
from pathlib import Path
from typing import Optional


def extract_horizon_belt(frame: np.ndarray, horizon_y: int, belt_half_height: int = 8) -> np.ndarray:
    h = frame.shape[0]
    y0 = max(0,     horizon_y - belt_half_height)
    y1 = min(h - 1, horizon_y + belt_half_height + 1)
    return frame[y0:y1, :, :]


def project_belt_to_line(belt: np.ndarray) -> np.ndarray:
    # Collapse belt height by averaging -> single pixel row
    return belt.mean(axis=0, keepdims=True).astype(np.uint8)


def estimate_horizon(frame: np.ndarray, method: str = "fixed_fraction", fraction: float = 0.45) -> int:
    h = frame.shape[0]
    if method == "fixed_fraction":
        return int(h * fraction)
    elif method == "vanishing":
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        if lines is None:
            return int(h * fraction)
        ys = []
        for rho, theta in lines[:, 0]:
            if abs(theta - np.pi / 2) < np.pi / 6:
                y = int(rho / np.sin(theta)) if np.sin(theta) != 0 else h // 2
                ys.append(y)
        return int(np.median(ys)) if ys else int(h * fraction)
    else:
        raise ValueError(f"Unknown method: {method}")


class MotionProfileBuilder:
    """Accumulates per-frame horizon lines into a 2D Motion Profile."""

    def __init__(self, width: int = 2592, max_frames: int = 1800):
        self.width      = width
        self.max_frames = max_frames
        self._lines: list = []
        self._timestamps: list = []

    def add_frame(self, frame: np.ndarray, horizon_y: int, timestamp: float = 0.0, belt_height: int = 8):
        belt = extract_horizon_belt(frame, horizon_y, belt_height)
        line = project_belt_to_line(belt)
        if line.shape[1] != self.width:
            line = cv2.resize(line[0], (self.width, 1))[np.newaxis, ...]
        self._lines.append(line)
        self._timestamps.append(timestamp)
        if len(self._lines) > self.max_frames:
            self._lines.pop(0)
            self._timestamps.pop(0)

    def get_profile(self):
        if not self._lines:
            raise ValueError("No frames added yet")
        return np.concatenate(self._lines, axis=0), list(self._timestamps)  # [T, W, 3]

    def get_patch(self, end_t: int, T: int = 256) -> np.ndarray:
        start = max(0, end_t - T + 1)
        lines = self._lines[start : end_t + 1]
        if len(lines) < T:
            lines = [np.zeros_like(lines[0])] * (T - len(lines)) + lines
        return np.concatenate(lines, axis=0)  # [T, W, 3]


class WidthProfileBuilder:
    """Builds WP(x,t) by painting vehicle bounding-box widths at the horizon row."""

    def __init__(self, width: int = 2592, max_frames: int = 1800):
        self.width      = width
        self.max_frames = max_frames
        self._lines: list = []

    def add_detections(self, detections: list, frame_width: int):
        # detections: list of (x_left, x_right) pixel ranges from YOLO
        line  = np.zeros((1, self.width), dtype=np.uint8)
        scale = self.width / frame_width
        for x_left, x_right in detections:
            xl, xr    = int(x_left * scale), int(x_right * scale)
            intensity = min(255, int((xr - xl) * 255 / self.width * 4))
            line[0, xl:xr] = intensity
        self._lines.append(line)
        if len(self._lines) > self.max_frames:
            self._lines.pop(0)

    def get_patch(self, end_t: int, T: int = 256) -> np.ndarray:
        if not self._lines:
            return np.zeros((T, self.width), dtype=np.uint8)
        start = max(0, end_t - T + 1)
        lines = self._lines[start : end_t + 1]
        if len(lines) < T:
            lines = [np.zeros_like(lines[0])] * (T - len(lines)) + lines
        return np.concatenate(lines, axis=0)  # [T, W]


def build_network_input(mp_patch: np.ndarray, wp_patch: np.ndarray, T: int = 256, W: int = 768) -> np.ndarray:
    # Resize and stack into 4-channel float32 array ready for ST2CN
    mp_resized = cv2.resize(mp_patch, (W, T))
    wp_resized = cv2.resize(wp_patch, (W, T))
    mp_float   = mp_resized.astype(np.float32) / 255.0
    wp_float   = wp_resized.astype(np.float32) / 255.0
    return np.concatenate([mp_float.transpose(2, 0, 1), wp_float[np.newaxis, ...]], axis=0)  # [4, T, W]


def process_video_to_profiles(
    video_path:       str,
    output_dir:       str,
    horizon_method:   str   = "fixed_fraction",
    horizon_fraction: float = 0.45,
    belt_height:      int   = 8,
    max_frames:       int   = 1800,
) -> tuple:
    """
    Process a video into MP and WP arrays and save as .npy.
    WP will be all zeros unless you hook in YOLO detections via add_detections().
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    mp_builder = MotionProfileBuilder(width=frame_w, max_frames=max_frames)
    wp_builder = WidthProfileBuilder(width=frame_w,  max_frames=max_frames)
    horizon_y: Optional[int] = None
    t = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if horizon_y is None:
            horizon_y = estimate_horizon(frame, horizon_method, horizon_fraction)
            print(f"Horizon at y={horizon_y}")
        mp_builder.add_frame(frame, horizon_y, timestamp=t, belt_height=belt_height)
        wp_builder.add_detections([], frame_width=frame_w)  # hook YOLO here
        t += 1.0 / src_fps

    cap.release()
    mp, _ = mp_builder.get_profile()
    wp    = (np.concatenate(wp_builder._lines, axis=0) if wp_builder._lines
             else np.zeros((len(mp_builder._lines), frame_w), dtype=np.uint8))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem
    np.save(out / f"{stem}_mp.npy", mp)
    np.save(out / f"{stem}_wp.npy", wp)
    print(f"Saved MP {mp.shape} and WP {wp.shape} to {output_dir}")
    return mp, wp


if __name__ == "__main__":
    # Quick sanity check with synthetic data
    W, T_total = 2592, 300
    builder = MotionProfileBuilder(width=W, max_frames=T_total)
    for i in range(T_total):
        fake = np.zeros((720, W, 3), dtype=np.uint8)
        builder.add_frame(fake, horizon_y=324)

    patch    = builder.get_patch(end_t=T_total - 1, T=256)
    wp_dummy = np.zeros((256, W), dtype=np.uint8)
    x        = build_network_input(patch, wp_dummy)
    assert x.shape == (4, 256, 768)
    print(f"4-channel input: {x.shape}  — check passed")
