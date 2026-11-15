"""Marker detection and generation that works on both OpenCV ArUco APIs.

Ubuntu 24.04 ships OpenCV 4.6, which has the function-style API (`detectMarkers`, `drawMarker`).
OpenCV 4.7+ replaced it with `ArucoDetector` and `generateImageMarker`.

The default marker family is AprilTag 25h9, read with a detector loosened for small and
blurred tags. Measured on rendered tags (40 per case, random rotation, noise):

    black square 48 px, blur sigma 3      default detector      loosened detector
    ArUco 4x4 (DICT_4X4_50)                     52 %                  92 %
    AprilTag 16h5                               92 %                  95 %, 5 % wrong ids
    AprilTag 25h9                               72 %                 100 %
    AprilTag 36h11                              42 %                  92 %

25h9 was also the only family with no wrong id in any case. Below 32 px with that blur
nothing reads: the black square of the tag has to be about 48 pixels wide in the picture.
Any other family of `cv2.aruco` (`DICT_4X4_50`, `DICT_APRILTAG_36h11`...) can still be named.

Those figures are for a recent OpenCV (5.0, `pip install opencv-contrib-python`). OpenCV 4.6,
the one in the dev container, is weaker on blurred tags: it reads the 48 px tag at blur sigma
1.5 (12 of 12) but not at sigma 3 (0 of 12), where it needs 64 px. Run the detection with a
recent OpenCV when the picture is blurred.
"""

import cv2
import numpy as np

DEFAULT_DICTIONARY = 'DICT_APRILTAG_25h9'


def get_dictionary(name: str = DEFAULT_DICTIONARY):
    dictionary_id = getattr(cv2.aruco, name)
    if hasattr(cv2.aruco, 'getPredefinedDictionary'):
        return cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.Dictionary_get(dictionary_id)


def detector_parameters():
    """Detector settings for small, blurred tags seen from far above: thresholding windows
    from 3 pixels up, small quads kept, corners refined to sub-pixel."""
    if hasattr(cv2.aruco, 'ArucoDetector'):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.minMarkerPerimeterRate = 0.01
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.perspectiveRemovePixelPerCell = 8
    return parameters


def detect_markers(image: np.ndarray, dictionary_name: str = DEFAULT_DICTIONARY) -> dict:
    """Return {marker id: 4x2 float array of corner pixels}.

    Corners are in the marker's own order: top-left, top-right, bottom-right, bottom-left, as
    seen when the marker is upright.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    dictionary = get_dictionary(dictionary_name)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(dictionary, detector_parameters())
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=detector_parameters()
        )
    if ids is None:
        return {}
    return {
        int(marker_id): np.asarray(quad, dtype=np.float64).reshape(4, 2)
        for marker_id, quad in zip(ids.flatten(), corners, strict=True)
    }


def generate_marker(
    marker_id: int, side_px: int, dictionary_name: str = DEFAULT_DICTIONARY
) -> np.ndarray:
    """Return a side_px x side_px grayscale image of the marker, without a white border."""
    dictionary = get_dictionary(dictionary_name)
    if hasattr(cv2.aruco, 'generateImageMarker'):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
    return cv2.aruco.drawMarker(dictionary, marker_id, side_px)
