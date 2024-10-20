from __future__ import annotations

import cv2 as cv
import numpy as np


def rotate(  # noqa: PLR0913
    image: np.ndarray,
    angle: float,
    *,
    wrap: bool = True,
    center: tuple[float, float] | None = None,
    scale: float = 1.0,
    fill: tuple = (255, 255, 255),
) -> np.ndarray:
    """
    Return `image` rotated by `angle` (in degress) in CCW direction.

    Args:
        image: numpy array
            Original image
        angle: float
            Angle (in degrees, counter-clockwise direction) to rotate
        wrap: bool
            Makes sure the original image is not clipped. Adds background as necessary.
        center: tuple
            (center_x, center_y)
        scale: float
            Resize scale
        fill: tuple: (B, G, R)
            Color to fill the extra backgrounds (if `wrap` is True)

    Returns:
        rotated_image: numpy array
            Rotated image

    """
    original_height, original_width = image.shape[:2]
    if center is None:
        center = (original_width / 2, original_height / 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, scale)

    if wrap:
        # grab the sin and cos (ie, the rotation components of the matrix)
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        # compute the new bounding dimensions of the image
        new_width = int((original_height * sin) + (original_width * cos))
        new_height = int((original_height * cos) + (original_width * sin))
        # adjust the rotation matrix to take into account translation
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
    else:
        new_height, new_width = original_height, original_width
    return cv.warpAffine(
        image,
        rotation_matrix,
        (new_width, new_height),
        borderValue=fill,
    )
