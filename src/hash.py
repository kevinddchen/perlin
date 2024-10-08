import hashlib
import math
import os

_MAX_UINT32 = 0xFFFFFFFF


def _hash_int(n: int) -> int:
    """
    Hash an integer to a uint32.
    """

    # cast integer to uint32, then convert to bytes. we use little-endian since
    # the least-significant-byte varies the most and should be in front.
    n_bts = int.to_bytes(n & _MAX_UINT32, length=4, byteorder="little")
    hash = hashlib.md5(n_bts).digest()
    return int.from_bytes(hash) & _MAX_UINT32


def _hash_combine(a: int, b: int) -> int:
    """
    Combine two uint32 hashes. Uses same method as boost::hash_combine.
    """

    a ^= (b + 0x9E3779B9 + ((a << 6) & _MAX_UINT32) + (a >> 2)) & _MAX_UINT32
    return a


def _hash_grid_point_md5(x: int, y: int, octave: int) -> int:
    """
    Hash grid point to a uint32, using MD5 hash.
    """

    if octave != 1:
        raise NotImplementedError("Octaves not implemented for MD5 yet")
    return _hash_combine(_hash_int(x), _hash_int(y))


def _hash_grid_point_fnv(x: int, y: int, octave: int) -> int:
    """
    Hash grid point to a uint32, using FNV-1a hash.
    """

    # cast integer to uint32, then convert to bytes. we use little-endian since
    # the least-significant-byte varies the most and should be in front.
    x_bts = int.to_bytes(x & _MAX_UINT32, length=4, byteorder="little")
    y_bts = int.to_bytes(y & _MAX_UINT32, length=4, byteorder="little")
    o_bts = int.to_bytes(octave & _MAX_UINT32, length=4, byteorder="little")

    # 32-bit variant of FNV-1a
    hash = 0x811C9DC5
    for char in x_bts + y_bts + o_bts:
        hash ^= char
        hash = (hash * 0x01000193) & _MAX_UINT32

    return hash


_hash_variant = os.getenv("PERLIN_HASH", "FNV")


def get_gradient_vector(x: int, y: int, octave: int = 1) -> tuple[float, float]:
    """
    Get random gradient vector for a given grid point. Set the `PERLIN_HASH`
    environment variable to pick between two hashing variants:

    - FNV: Use FNV-1a hash. This is the default
    - MD5: Use MD5 hash.
    """

    if _hash_variant == "FNV":
        h = _hash_grid_point_fnv(x, y, octave)
    elif _hash_variant == "MD5":
        h = _hash_grid_point_md5(x, y, octave)
    else:
        raise NotImplementedError(f"Hashing not implemented: {_hash_variant}")

    angle = 2 * math.pi * h / (_MAX_UINT32 + 1)
    return math.cos(angle), math.sin(angle)
