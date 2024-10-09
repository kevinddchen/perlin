from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import tyro

from perlin import RenderOpts, render


def main(
    opts: RenderOpts,
    out_path: Path = Path("perlin.png"),
    cmap: Literal["bwr", "gray"] = "gray",
    interpolation: Literal["nearest", "bilinear"] = "nearest",
) -> None:
    """
    Render Perlin noise.

    Args:
        opts: Perlin noise rendering options.
        out_path: path to output image file.
        cmap: color map for the image.
        interpolation: used for displaying the image.
    """
    
    arr = render(opts)
    plt.imshow(arr, cmap=cmap, interpolation=interpolation)
    plt.savefig(out_path)


if __name__ == "__main__":
    tyro.cli(main)
