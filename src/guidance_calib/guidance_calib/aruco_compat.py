"""ArUco detection and generation that works on both OpenCV ArUco APIs.

Ubuntu 24.04 ships OpenCV 4.6, which has the function-style API (`detectMarkers`, `drawMarker`).
OpenCV 4.7+ replaced it with `ArucoDetector` and `generateImageMarker`.
"""

import cv2
import numpy as np

DEFAULT_DICTIONARY = 'DICT_4X4_50'


def get_dictionary(name: str = DEFAULT_DICTIONARY):
    dictionary_id = getattr(cv2.aruco, name)
    if hasattr(cv2.aruco, 'getPredefinedDictionary'):
        return cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.Dictionary_get(dictionary_id)


def detect_markers(image: np.ndarray, dictionary_name: str = DEFAULT_DICTIONARY) -> dict:
    """Return {marker id: 4x2 float array of corner pixels}.

    Corners are in the marker's own order: top-left, top-right, bottom-right, bottom-left, as
    seen when the marker is upright.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    dictionary = get_dictionary(dictionary_name)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
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
