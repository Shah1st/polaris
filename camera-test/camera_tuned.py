import cv2
import numpy as np
import pyrealsense2 as rs

# =====================================================================
# SETTINGS - change these and re-run to adjust the image
# =====================================================================
# Must be a mode listed by check.py; depth and color must match on the D405
WIDTH, HEIGHT, FPS = 848, 480, 30

# --- Depth display ---
MIN_DIST_M = 0.07  # closer than this is shown black (D405 min is ~7 cm)
MAX_DIST_M = 0.50  # farther than this is shown black; colors are stretched over MIN..MAX
COLORMAP = cv2.COLORMAP_JET  # try COLORMAP_TURBO, COLORMAP_INFERNO, COLORMAP_BONE
NEAR_IS_RED = True  # False -> near = blue

# --- Depth cleanup ---
ALIGN_DEPTH_TO_COLOR = True  # depth pixels line up with color pixels
USE_SMOOTHING = True  # spatial + temporal filters (less speckle, slight lag)
FILL_HOLES = False  # fill black gaps with neighbouring values

# --- Camera exposure (hardware) ---
AUTO_EXPOSURE = True
EXPOSURE_US = 8000  # only used when AUTO_EXPOSURE = False (microseconds)
GAIN = 16  # only used when AUTO_EXPOSURE = False

# --- Color image tweaks (software, OpenCV) ---
CONTRAST = 1.0  # 1.0 = unchanged, >1 more contrast
BRIGHTNESS = 0  # -100..100
FLIP = False  # True to mirror the image horizontally

# --- Distance grid overlay ---
SHOW_GRID = True
GRID_ROWS, GRID_COLS = 8, 12  # test grid - raise for more precision (text shrinks to fit)
# Median over a (2*PATCH+1)^2 patch around each cell centre; 0 = exactly the centre pixel
GRID_PATCH = 2
GRID_COLOR = (255, 255, 255)  # BGR line colour
# =====================================================================


def set_option(sensor, option, value):
    """Set an option if the sensor supports it, clamped to its valid range."""
    if sensor.supports(option):
        r = sensor.get_option_range(option)
        sensor.set_option(option, float(min(max(value, r.min), r.max)))


def put_label(img, text, center, scale=0.6):
    """Draw text centred on a point, with a dark outline so it reads on any background."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    org = (int(center[0] - tw / 2), int(center[1] + th / 2))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def draw_distance_grid(images, depth_m, rows, cols, patch):
    """Split the image into rows x cols cells and show the distance at each cell centre."""
    h, w = depth_m.shape
    cell_w, cell_h = w / cols, h / rows
    # pick a font size so a label like "88.8cm" fills ~75% of a cell's width
    (ref_w, _), _ = cv2.getTextSize('88.8cm', cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    scale = min(0.6, 0.75 * cell_w / ref_w)

    for img in images:
        for c in range(1, cols):
            x = int(c * cell_w)
            cv2.line(img, (x, 0), (x, h - 1), GRID_COLOR, 1)
        for r in range(1, rows):
            y = int(r * cell_h)
            cv2.line(img, (0, y), (w - 1, y), GRID_COLOR, 1)

    for r in range(rows):
        for c in range(cols):
            cx, cy = int((c + 0.5) * cell_w), int((r + 0.5) * cell_h)
            region = depth_m[
                max(cy - patch, 0) : cy + patch + 1, max(cx - patch, 0) : cx + patch + 1
            ]
            valid = region[region > 0]
            text = f'{np.median(valid) * 100:.1f}cm' if valid.size else '--'
            for img in images:
                cv2.circle(img, (cx, cy), 4, (0, 0, 0), -1)  # marks the sampled pixel
                cv2.circle(img, (cx, cy), 2, (255, 255, 255), -1)
                put_label(img, text, (cx, cy + int(cell_h * 0.25)), scale)


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

print('[INFO] Starting streaming...')
profile = pipeline.start(config)
device = profile.get_device()

# Depth units: D405 uses 0.1 mm per unit by default (D435 uses 1 mm), so always read it.
depth_scale = device.first_depth_sensor().get_depth_scale()
print(f'[INFO] Depth scale: {depth_scale} m per unit')

# Exposure / gain (on the D405 these live on the same sensor as depth)
for s in device.query_sensors():
    set_option(s, rs.option.enable_auto_exposure, 1 if AUTO_EXPOSURE else 0)
    if not AUTO_EXPOSURE:
        set_option(s, rs.option.exposure, EXPOSURE_US)
        set_option(s, rs.option.gain, GAIN)

# Processing blocks
align = rs.align(rs.stream.color)
threshold = rs.threshold_filter(MIN_DIST_M, MAX_DIST_M)
to_disparity = rs.disparity_transform(True)
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
to_depth = rs.disparity_transform(False)
holes = rs.hole_filling_filter()

try:
    while True:
        frames = pipeline.wait_for_frames()
        if ALIGN_DEPTH_TO_COLOR:
            frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # ---- clean up depth ----
        depth = threshold.process(depth_frame)
        if USE_SMOOTHING:
            depth = to_disparity.process(depth)
            depth = spatial.process(depth)
            depth = temporal.process(depth)
            depth = to_depth.process(depth)
        if FILL_HOLES:
            depth = holes.process(depth)

        depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * depth_scale
        color = np.asanyarray(color_frame.get_data())

        # ---- depth -> colored image, stretched over MIN..MAX ----
        valid = (depth_m > MIN_DIST_M) & (depth_m < MAX_DIST_M)
        scaled = np.zeros(depth_m.shape, dtype=np.uint8)
        span = MAX_DIST_M - MIN_DIST_M
        scaled[valid] = (255 * (depth_m[valid] - MIN_DIST_M) / span).astype(np.uint8)
        if NEAR_IS_RED:
            scaled[valid] = 255 - scaled[valid]
        depth_vis = cv2.applyColorMap(scaled, COLORMAP)
        depth_vis[~valid] = 0  # no data / out of range -> black

        # ---- color tweaks ----
        color = cv2.convertScaleAbs(color, alpha=CONTRAST, beta=BRIGHTNESS)
        if FLIP:
            color = cv2.flip(color, 1)
            depth_vis = cv2.flip(depth_vis, 1)

        # ---- distance overlay ----
        if FLIP:
            depth_m = cv2.flip(depth_m, 1)  # keep numbers matching the flipped images
        if SHOW_GRID:
            draw_distance_grid((color, depth_vis), depth_m, GRID_ROWS, GRID_COLS, GRID_PATCH)
        else:
            cy, cx = depth_m.shape[0] // 2, depth_m.shape[1] // 2
            d = depth_m[cy, cx]
            label = f'{d * 100:.1f} cm' if d > 0 else 'no data'
            for img in (color, depth_vis):
                cv2.drawMarker(img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
                put_label(img, label, (cx, cy - 20), 0.7)

        cv2.imshow('D405 Color', color)
        cv2.imshow('D405 Depth', depth_vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # q or Esc to quit
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
