"""
Data plotting functionalities.
"""
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Set font
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
# Solve the problem of negative sign in chinese
plt.rcParams["axes.unicode_minus"] = False


def geo_cm(gamma=0.6):
    """
    :param gamma: float
    :return: `matplotlib` colormap
    """
    color_list = ["#0000ff", "#ffffff", "#FFD400", "#CD5555"]
    return LinearSegmentedColormap.from_list("geo_cm", color_list, gamma=gamma)


def plot_log_curve(x, y, **kwargs):
    """
    :param x: `np.array`, typically rho values, or other earth physics properties.
    :param y: `np.array`, typically depth values.
    :param kwargs: dict, `plt.plot()` options
    :return: `plt.Axes`
    """
    ax = plt.figure().add_subplot()
    ax.plot(x, y, **kwargs)
    return ax


def show_values(val1, val2, titles, **kwargs):
    """
    :param val1: ndarray;
    :param val2: ndarray;
    :param titles: list, [title1, title2];
    :param kwargs: dict, `plt.imshow()` options;
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
    img1 = ax1.imshow(val1, **kwargs)
    fig.colorbar(img1, ax=ax1)
    img2 = ax2.imshow(val2, **kwargs)
    fig.colorbar(img2, ax=ax2)
    ax1.set_title(titles[0])
    ax2.set_title(titles[1])
    plt.show()


def add_topography(topo, ax, **kwargs):
    p = plt.Polygon(topo, **kwargs)
    ax.add_artist(p)
    return ax


def add_layers(x, layers, ax, **kwargs):
    assert isinstance(layers, dict)
    for label, y in layers.items():
        ax.plot(x, y, label=label, **kwargs)
    return ax


def section_contour(x, y, rho,
                    topo=None,
                    layers=None,
                    save_img=False,
                    img_name=None,
                    **kwargs):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=80)
    img = ax.contourf(x, y, rho, cmap=geo_cm(), **kwargs)
    plt.colorbar(img, ax=ax, label=r"Resistivity ($\Omega\cdot m$)")
    ax.contour(x, y, rho, colors='k', linewidths=0.5, **kwargs)
    if topo is not None:
        ax = add_topography(topo, ax, fc="white", ec=None)
    if layers is not None:
        ax = add_layers(x, layers, ax)
        ax.legend()
    ax.spines["top"].set_visible(False)
    if save_img:
        if img_name is None:
            img_name = 'output.png'
        fig.tight_layout()
        fig.savefig(img_name)
    else:
        return ax
