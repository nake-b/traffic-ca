import random

import matplotlib.animation as animation
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np


def random_bool(p: float) -> bool:
    return random.random() < p


def plot1d_animate(
    ca,
    title="",
    *,
    colormap="Greys",
    show_grid=True,
    show_margin=True,
    scale=0.6,
    dpi=80,
    interval=50,
    save=False,
    autoscale=False,
    show=True,
    **imshow_kwargs
):
    """
    Animate the given 1D cellular automaton.

    The `show_margin` argument controls whether or not a margin is displayed in the resulting plot. When `show_margin`
    is set to `False`, then the plot takes up the entirety of the window. The `scale` argument is only used when the
    `show_margins` argument is `False`. It controls the resulting scale (i.e. relative size) of the image when there
    are no margins.

    The `dpi` argument represents the dots per inch of the animation when it is saved. There will be no visible effect
    of the `dpi` argument if the animation is not saved (i.e. when `save` is `False`).

    :param ca:  the 2D cellular automaton to animate

    :param title: the title to place on the plot (default is "")

    :param colormap: the color map to use (default is "Greys")

    :param show_grid: whether to display a grid (default is False)

    :param show_margin: whether to display the margin (default is True)

    :param scale: the scale of the figure (default is 0.6)

    :param dpi: the dots per inch of the image (default is 80)

    :param interval: the delay between frames in milliseconds (default is 50)

    :param save: whether to save the animation to a local file (default is False)

    :param autoscale: whether to autoscale the images in the animation; this should be set to True if the first
                      frame has a uniform value (e.g. all zeroes) (default is False)

    :param show: show the plot (default is True)

    :param imshow_kwargs: keyword arguments for the Matplotlib `imshow` function

    :return: the animation
    """
    cmap = plt.get_cmap(colormap)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_yticklabels([])
    ax.set_yticks([])
    plt.title(title)
    if not show_margin:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    grid = _add_grid_lines(ca, ax, show_grid)

    im = plt.imshow(
        _get_img_array_from_1d_ca(ca[0]), animated=True, cmap=cmap, **imshow_kwargs
    )
    if not show_margin:
        baseheight, basewidth = im.get_size()
        fig.set_size_inches(basewidth * scale, baseheight * scale, forward=True)

    i = {"index": 0}

    def updatefig(*args):
        i["index"] += 1
        if i["index"] == len(ca):
            i["index"] = 0
        im.set_array(_get_img_array_from_1d_ca(ca[i["index"]]))
        if autoscale:
            im.autoscale()
        return im, grid

    ani = animation.FuncAnimation(
        fig, updatefig, interval=interval, blit=True, save_count=len(ca)
    )
    if save:
        ani.save("evolved.gif", dpi=dpi, writer="imagemagick")
    if show:
        plt.show()
    return ani


def _add_grid_lines(ca, ax, show_grid):
    """
    Adds grid lines to the plot.

    :param ca: the 2D cellular automaton to plot

    :param ax: the Matplotlib axis object

    :param show_grid: whether to display the grid lines

    :return: the grid object
    """
    grid_linewidth = 0.0
    if show_grid:
        grid_linewidth = 0.5
    vertical = np.arange(-0.5, len(ca[0]), 1)
    horizontal = np.arange(-0.5, 0, 1)
    lines = [[(x, y) for y in (-0.5, horizontal[-1])] for x in vertical] + [
        [(x, y) for x in (-0.5, vertical[-1])] for y in horizontal
    ]
    grid = mcoll.LineCollection(
        lines, linestyles="-", linewidths=grid_linewidth, color="grey"
    )
    ax.add_collection(grid)

    return grid


def _get_img_array_from_1d_ca(ca: np.ndarray) -> np.ndarray:
    return np.array([(x, 0) for x in ca]).T
