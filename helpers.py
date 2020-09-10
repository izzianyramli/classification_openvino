import numpy as np
import base64
import sys


def base64_encode_image(a):
    return base64.b64encode(a).decode("utf-8")


def base64_decode_image(a, dtype, shape):
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")
    a = np.frombuffer(base64.decodebytes(a), dtype=dtype)
    a = a.reshape(shape)
    return a