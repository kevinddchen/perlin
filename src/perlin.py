import math

from .hash import get_gradient_vector


def _grid_dot_product(grid_x: int, grid_y: int, x: float, y: float) -> float:
    """
    Compute contribution to the Perlin noise corresponding to the dot product
    between the random gradient vector at the grid point and the displacement
    vector to the point.

    Args:
        grid_x: x coordinate of grid point.
        grid_y: y coordinate of grid point.
        x: x coordinate of point.
        y: y coordinate of point.

    Returns:
        Dot product value, normalized to [-1, 1].
    """

    grad_x, grad_y = get_gradient_vector(grid_x, grid_y)
    disp_x = x - grid_x
    disp_y = y - grid_y
    dot = grad_x * disp_x + grad_y * disp_y

    # In 2d, dot is in [-1/sqrt(2), 1/sqrt(2)]. So normalize by multipling by sqrt(2).
    return math.sqrt(2) * dot


def _interpolate(a0: float, a1: float, t: float) -> float:
    """
    Interpolate between a0 and a1, where t is in [0, 1]. Uses "smootherstep"
    function, i.e. first and second derivatives vanish at endpoints.
    """

    return (a1 - a0) * t * t * t * (t * (6 * t - 15) + 10) + a0


def perlin(x: float, y: float) -> float:
    """Returns Perlin noise at point, normalized to [-1, 1]."""

    # determine grid points
    x0 = int(x)
    y0 = int(y)

    # compute contributions from each grid point
    bottom_left = _grid_dot_product(x0, y0, x, y)
    bottom_right = _grid_dot_product(x0 + 1, y0, x, y)
    top_right = _grid_dot_product(x0 + 1, y0 + 1, x, y)
    top_left = _grid_dot_product(x0, y0 + 1, x, y)

    # interpolation weights
    t_x = x - x0
    t_y = y - y0

    # interpolate value
    bottom = _interpolate(bottom_left, bottom_right, t_x)
    top = _interpolate(top_left, top_right, t_x)
    value = _interpolate(bottom, top, t_y)

    return value
