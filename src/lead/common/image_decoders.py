"""The camera image decoders behind the py123d log reads.

Both return what py123d's own decoders return: grayscale as ``(H, W)``, RGB
as ``(H, W, 3)``, 16-bit PNG as uint16. The turbojpeg handle is created
lazily because it loads the libturbojpeg system library, which only runs
that decode camera JPEGs need installed.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import imagecodecs
import numpy as np


@functools.cache
def _turbojpeg_decoder() -> Callable[[bytes], np.ndarray]:
    from turbojpeg import TJPF_RGB, TurboJPEG

    turbo = TurboJPEG()
    return lambda jpeg_bytes: turbo.decode(jpeg_bytes, pixel_format=TJPF_RGB)


def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """Decode an RGB camera JPEG via libjpeg-turbo.

    Args:
        jpeg_bytes: The encoded JPEG payload.

    Returns:
        The image as ``(H, W, 3)`` uint8.
    """
    return _turbojpeg_decoder()(jpeg_bytes)


def decode_png(png_bytes: bytes) -> np.ndarray:
    """Decode a PNG via imagecodecs.

    Args:
        png_bytes: The encoded PNG payload.

    Returns:
        The image, grayscale as ``(H, W)`` uint8 or uint16, RGB as ``(H, W, 3)``.
    """
    return imagecodecs.png_decode(png_bytes)
