import uuid
import numpy as np


def is_int(n):
    return isinstance(n, (int, np.integer))


def is_float(n):
    return isinstance(n, (float, np.float))


def is_uuid(n):
    return isinstance(n, uuid.UUID)


def is_bool(n):
    return isinstance(n, (bool, np.bool_))


def is_array(n):
    return isinstance(n, np.ndarray)