import ezdxf
import numpy as np
import welly
from striplog import Legend, Striplog

from .plotting import plot_log_curve


def get_logging_entity_from_dxf(dxf_file, obj_type='POLYLINE'):
    """
    :param dxf_file: filename of the ASCII- or Binary DXF document encoding
    :param obj_type: str, the type of dxf object we need to find in the DXF document
    :return: well logging DXF obj
    """
    dxf = ezdxf.readfile(dxf_file)
    model_space = dxf.modelspace()
    for obj in model_space:
        if obj.dxftype() == obj_type:
            print("Well logging on layer: %s\n" % obj.dxf.layer)
            return obj


def get_logging_data(dxf_obj,
                     x_scale=1,
                     y_scale=1,
                     delta_x=0,
                     delta_y=0,
                     enable_plotting=False):
    pts = []
    for p in dxf_obj.points():
        pts.append(list(p.xyz))
    pts = np.array(pts)
    x = pts[:, 0] * x_scale + delta_x
    y = pts[:, 1] * y_scale + delta_y
    if enable_plotting:
        ax = plot_log_curve(x, y)
        ax.set(xlabel=r"Resistivity ($\Omega\cdot m$)",
               ylabel=r"Altitude (m)")
    return np.c_[x, y]


def welly_curve(data, index, logging_type):
    """
    :param data: 1D/2D/3D curve numerical or categorical data. Dict can contain Series, arrays, constants, dataclass
    or list-like objects. If data is a dict, column order follows insertion-order. Input is passed as 'data' argument
     of pd.DataFrame constructor.
    :param index: Optional. Index to use for resulting pd.DataFrame. Will default to RangeIndex if no indexing
    information part of input data and no index provided. Input is passed to 'index' parameter of the pd.DataFrame
    constructor.
    :param logging_type: Optional. The mnemonic(s) of the curve if the data does not have them. It is passed as the
    'columns' parameter of pd.DataFrame constructor. Single mnemonic for 1D data, multiple mnemonics for 2D data.
    :return: `welly.Curve`
    """
    return welly.Curve(data=data, index=index, mnemonic=logging_type, index_name='depth')


def strip_log(csv_text, legend_text, show_result=False):
    strip = Striplog.from_csv(text=csv_text)
    legend = Legend.from_csv(text=legend_text)
    if show_result:
        strip.plot(legend, aspect=5)
    else:
        return strip, legend
