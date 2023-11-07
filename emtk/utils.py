import numpy as np
import pygimli as pg


def get_vals_from_model(mesh: pg.core.Mesh, model, pos, delta=None):
    """
    Parameters
    ----------
    mesh: pg.core.Mesh
    model: ndarray
        The property values on mesh.
    pos: int or float
        The x coordinate value in the mesh range
    delta: int or float
        The vertical step in y-direction.
    Returns
    -------
        ndarray: the property value in y-direction.
    """

    x_min, x_max, y_min, y_max = (
        mesh.xmin(),
        mesh.xmax(),
        mesh.ymin(),
        mesh.ymax(),
    )
    if pos < x_min or pos > x_max:
        raise RuntimeError("The x out of mesh domain in x-direction!")
    if delta is None:
        delta_y = -20.
    else:
        if delta > 0:
            delta_y = -delta
        else:
            delta_y = delta
    assert isinstance(delta_y, float or int)
    y = np.arange(y_max, y_min, delta_y)
    x = np.ones_like(y) * pos
    pos_vec = [pg.Pos(p) for p in zip(x, y)]
    out = pg.interpolate(srcMesh=mesh, inVec=model, destPos=pos_vec)
    return np.c_[y, out]
