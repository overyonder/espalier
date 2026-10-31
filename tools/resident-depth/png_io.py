"""libpng decodes sample bytes into one reusable, vertically addressed buffer."""

import ctypes as C
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_library = C.CDLL(
    os.environ.get(
        "RESIDENT_VULKAN_LIBRARY",
        str(ROOT / "Intermediate/resident-depth/resident_vk.so"),
    )
)
_library.rd_decode_png.argtypes = [C.c_char_p, C.c_void_p, C.c_uint32]
_library.rd_decode_png.restype = C.c_int
_library.rd_error.restype = C.c_char_p


class Decoder:
    def __init__(self, width):
        self.width = width
        self.buffer = (C.c_ubyte * (width * width * 4))()
        self.pixels = np.ctypeslib.as_array(self.buffer).reshape(width, width, 4)

    def decode(self, path):
        self.pixels.fill(0)
        self.pixels[..., 3] = 255
        if (
            _library.rd_decode_png(
                os.fsencode(path), C.addressof(self.buffer), self.width
            )
            < 0
        ):
            raise ValueError(_library.rd_error().decode())
        return self.buffer
