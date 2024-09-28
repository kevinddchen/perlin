import hashlib
import math

_HASH_SIZE = 2**32


def _hash_int(n: int) -> int:
    """
    Hash an integer to a uint32.
    """

    hash = hashlib.md5(int.to_bytes(n))
    return int.from_bytes(hash.digest()) % _HASH_SIZE


def _hash_combine(a: int, b: int) -> int:
    """
    Combine two uint32 hashes. Uses same method as boost::hash_combine.
    """

    a ^= (b + 0x9E3779B9 + (a << 6) % _HASH_SIZE + (a >> 2)) % _HASH_SIZE
    return a


def _hash_grid_point(x: int, y: int) -> tuple[float, float]:
    """
    Hash grid point to a unit vector.
    """

    hash = _hash_combine(_hash_int(x), _hash_int(y))
    angle = (hash / _HASH_SIZE) * math.pi
    return math.cos(angle), math.sin(angle)


def _compute_dot_product(grid_x: int, grid_y: int, x: float, y: float) -> float:
    """
    Args:
        grid_x: x coordinate of grid point.
        grid_y: y coordinate of grid point.
        x: x coordinate of point.
        y: y coordinate of point.

    Returns:
        Contribution to Perlin noise corresponding to the grid point.
    """

    grad_x, grad_y = _hash_grid_point(grid_x, grid_y)
    disp_x = x - grid_x
    disp_y = y - grid_y
    return grad_x * disp_x + grad_y * disp_y


def _interpolate(x0: float, x1: float, t: float) -> float:
    """
    Interpolate between x0 and x1, where t is in [0, 1].
    """

    # use smoothstep
    return (x1 - x0) * (3 - 2 * t) * t * t + x0


def perlin(x: float, y: float) -> float:
    """Returns Perlin noise at point, normalized to [-1, 1]."""

    # determine grid points
    x0 = int(x)
    y0 = int(y)

    # compute contributions from each grid point
    bottom_left = _compute_dot_product(x0, y0, x, y)
    bottom_right = _compute_dot_product(x0 + 1, y0, x, y)
    top_right = _compute_dot_product(x0 + 1, y0 + 1, x, y)
    top_left = _compute_dot_product(x0, y0 + 1, x, y)

    # interpolation weights
    t_x = x - x0
    t_y = y - y0

    # interpolate value
    bottom = _interpolate(bottom_left, bottom_right, t_x)
    top = _interpolate(top_left, top_right, t_x)
    value = _interpolate(bottom, top, t_y)

    return value
