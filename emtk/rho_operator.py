"""
Rho data operator, including geophysical section and target layer operators.
"""
import re

import gstools as gs
import numpy as np
import pandas as pd
from pandas import DataFrame
from pykrige import kriging_tools as kt
from pykrige.ok import OrdinaryKriging
from scipy.ndimage import gaussian_filter

from .plotting import show_values


def calculate_the_index(grid_x, grid_y, x, y):
    x_idx = np.where(grid_x == x)[0][0]
    t = np.abs(grid_y - y)
    y_idx = np.where(t == np.min(t))[0][0]
    return x_idx, y_idx


global xx, yy, kriging_data, var


def kriging_obj_operator(
        kriging_obj,
        show_kriging_result=False,
        use_filter=False,
        filter_sigma=2,
        show_filter_result=False,
        save_kriging_data=False,
        f_name=None,
        verbose=False,
):
    """Kriging object operator
    :param kriging_obj: dict,
                         {'method': 'Section' or 'Layer',
                          'obj': `SectionToolkit.section_rho_kriging` or `LayerToolkit.layer_rho_kriging`,
                          'layer_label': str, layer label. Only usefully when 'method' is 'Layer'.}
    :param show_kriging_result: show_kriging_result: default `False`
    :param use_filter: Whether to use gaussian filter to kriging result, provided by `scipy.ndimage`, default `False`.
    :param filter_sigma: Standard deviation for Gaussian kernel, only useful when `use_filter=True`.
    :param show_filter_result: Show the comparison between kriging result and image filter result.
    :param save_kriging_data: Whether to write gridded data to ASCII grid file in zmap format, default `False`.
    :param f_name: Name of '*.zmap' file. Default name is 'output.zmap'. Only useful when `save_kriging_data=True`.
    :param verbose: bool, whether to visualize the progress. Default `False`.
    :return: x : numpy array, shape (N, ) 1D array of N X-coordinates (horizontal direction).
             y : numpy array, shape (M, ) 1D array of M y-coordinates (vertical direction).
             grid_array : numpy array, shape (M, N) (M, N) array of grid values, where M is number
             of Y-coordinates and N is number of X-coordinates. The array entry corresponding to the lower-left
             coordinates is at index [M, 0], so that the array is oriented as it would be in X-Z space.
    """
    global xx, yy, kriging_data, var
    if kriging_obj["method"] == "Section":
        xx, yy, kriging_data, var = kriging_obj["obj"].section_rho_kriging(
            log_scale=True
        )
    elif kriging_obj["method"] == "Layer":
        xx, yy, kriging_data, var = kriging_obj["obj"](
            layer_label=kriging_obj["layer_label"]
        )
    else:
        raise KeyError(
            "The setting of `kriging_obj` seems to be incorrect, please check!"
        )
    if show_kriging_result:
        show_values(kriging_data, var, ["Kriging rho", "Variance"], cmap="RdBu_r")
    if use_filter:
        filter_data = gaussian_filter(kriging_data, sigma=filter_sigma)
        if show_filter_result:
            show_values(
                kriging_data,
                filter_data,
                ["Kriging rho", "Filter kriging rho"],
                cmap="RdBu_r",
            )
        else:
            if save_kriging_data:
                if f_name is None:
                    f_name = "output.zmap"
                kt.write_zmap_grid(xx, yy, filter_data, filename=f_name)
                if verbose:
                    print(
                        "Filtered kriging rho data have saved successfully: " + f_name
                    )
            else:
                return xx, yy, filter_data
    else:
        if save_kriging_data:
            if f_name is None:
                f_name = "output.zmap"
            kt.write_zmap_grid(xx, yy, kriging_data, filename=f_name)
            if verbose:
                print("Kriging rho data have saved successfully: " + f_name)
        else:
            print("There is none we have done in this operation!")


class SectionToolkit:
    def __init__(
            self,
            section_index,
            rho_dir,
            z_range=None,
            rho_fmt=".dat",
            topo=None,
            region_layer=None,
            target_labels=None,
    ):
        """ Section toolkit used to operate rho file
        :param section_index: int, line index
        :param rho_dir: str, rho file folder where rho file store at
        :param z_range: list or tuple, limitation of z-direction, e.g. [z_min, z_max]
        :param rho_fmt: str, rho file extension name, default '*.dat'
        :param topo: str, region topography file, kriging data with format (`*.zmap`)
        :param region_layer: str, region_layer file (*.csv)

        """
        self.line_index = section_index
        self.rho_folder = rho_dir
        self.rho_file = self.rho_folder + str(self.line_index) + rho_fmt
        if z_range is not None:
            self.z_range = z_range
        else:
            self.z_range = [-800, 100]
        self.topography_file = topo
        self.layer_file = region_layer
        if target_labels is not None:
            self.target_layer_labels = target_labels
        else:
            self.target_layer_labels = None

    @property
    def rho_data(self):
        """
        :return: rho data type is `np.array`
        """
        res_file = self.rho_file
        f = open(res_file).read().split("\n")
        res = f[2:]
        dist, elevation, rho = [], [], []
        pattern = re.compile(r"[+-]*\d+\.?\d*[Ee]*[+-]*\d+|\d+")
        for i in range(len(res)):
            results = pattern.findall(res[i])
            if len(results) == 3:
                dist.append(float(results[0]))
                elevation.append(float(results[1]))
                rho.append(float(results[2]))
            else:
                continue
        return np.c_[dist, elevation, rho]

    @property
    def rho_statistic(self):
        """
        :return: describe statistic of rho data
        """
        df = pd.DataFrame(self.rho_data, columns=["x", "z", "rho"])
        return df.describe()

    @property
    def topography(self):
        """
        :return: the topography of the specific line
        """
        altitudes, lines, points = kt.read_zmap_grid(self.topography_file)[:3]
        pts = self.domain[0]
        p_min, p_max = min(pts), max(pts)
        l_ind = np.where(lines == self.line_index)[0][0]
        tmp = altitudes[:, l_ind]
        p_ind = (points >= p_min) & (points <= p_max)
        x = points[p_ind]
        x = np.append(x, [p_max, p_min, p_min])
        z = tmp[p_ind]
        z = np.append(z, [self.z_range[1], self.z_range[1], z[0]])
        return np.c_[x, z]

    @property
    def domain(self):
        """
        :return: section domain limitation
        """
        z_min, z_max = self.z_range
        data = self.rho_data
        x = np.unique(data[:, 0])
        assert np.unique(np.diff(x)).size == 1
        delta = np.diff(x)[0]
        z = np.arange(z_min, z_max + delta, delta, dtype=np.float64)
        return x, z

    @property
    def target_layers_on_section(self):
        try:
            assert isinstance(self.target_layer_labels, list)
        except AssertionError:
            print("Please assign to `target_layer_labels` variable correctly.")
        layers = pd.read_csv(self.layer_file)
        columns = list(layers.columns)[:2] + self.target_layer_labels
        # target_layer = layers[columns].values
        target_layer = layers[columns]
        line_layer = target_layer[target_layer["line"] == self.line_index].values
        # l_ind = target_layer[:, 1] == self.line_index
        # line_layer = target_layer[l_ind]
        pts = self.domain[0]
        p_min, p_max = min(pts), max(pts)
        p_ind = (line_layer[:, 1] >= p_min) & (line_layer[:, 1] <= p_max)
        layer = line_layer[p_ind][:, 2:]
        return dict(zip(self.target_layer_labels, layer.T))

    def section_rho_kriging(self, top_rho=40, rho_statistic=False, log_scale=False):
        """
        :param top_rho: the top limitation of rho value
        :param rho_statistic: show rho data describe statistic result
        :param log_scale: `np.log10(val)`, default False
        :return:
         x : numpy array, shape (N, ) 1D array of N X-coordinates (horizontal direction).
         z : numpy array, shape (M, ) 1D array of M z-coordinates (vertical direction).
         kriging_data: (grid_array) data of specified grid or at the specified set of points.
         numpy array, shape (M, N) (M, N) array of grid values, where M is number of Y-coordinates
         and N is number of X-coordinates. The array entry corresponding to the lower-left
         coordinates is at index [M, 0], so that the array is oriented as it would be in X-Z space.
         var: Variance at specified grid points or at the specified set of points
        """
        data = self.rho_data
        if rho_statistic:
            print(self.rho_statistic)
        ind = data[:, 2] < top_rho
        data = data[ind]
        x, y = self.domain
        delta = np.diff(x)[0]
        var_model = gs.Exponential(dim=2, len_scale=delta * 2, var=0.5)
        ok2d = OrdinaryKriging(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            variogram_model=var_model,
            verbose=0,
            enable_plotting=False,
        )
        val, sigma_sq = ok2d.execute("grid", x, y)
        if log_scale:
            val = np.log10(val) * 10
        return x, y, val, sigma_sq


class LayerToolkit:
    def __init__(self, formation, labels, layer_domain, sec_kriging_files_dir):
        """
        :param formation: `pd.DataFrame`
        :param labels: list, layers label
        :param layer_domain: list, [x_min, x_max, y_min, y_max], region domain
        :param sec_kriging_files_dir: str, line section kriging files folder, prefer relative direction
        """
        self.formation = formation
        self.labels = labels
        self.layer_domain = layer_domain
        self.kriging_files_dir = sec_kriging_files_dir

    delta = 20

    def extract_layer_rho(self, layer_label, verbose=False, save_opts=None):
        """
        :param layer_label: str.
        :param verbose: bool, Whether to visualize the progress, default `False`.
        :param save_opts: dict, {'save': 0 or 1, 'f_name': str, filename}
        :return: layer rho, `pd.DataFrame`
        """
        formation = self.formation
        filenames = list(formation["line"].drop_duplicates())
        d = {"line": [], "point": [], "rho": []}
        for i in range(len(filenames)):
            filename = str(filenames[i]) + ".zmap"
            try:
                data, grid_x, grid_y = kt.read_zmap_grid(
                    self.kriging_files_dir + filename
                )[:3]
                if verbose:
                    print(filename + " has been loaded successfully!")
            except FileNotFoundError:
                if verbose:
                    print(filename + " not exists, ignore it!")
                continue
            sec_data = formation[formation["line"] == filenames[i]]
            sec_data = sec_data.reset_index()
            for x in grid_x:
                idx = sec_data[sec_data["point"] == x].index[0]
                y = sec_data.iloc[idx][layer_label]
                x_idx, y_idx = calculate_the_index(grid_x, grid_y, x, y)
                d["line"].append(filenames[i])
                d["point"].append(x)
                d["rho"].append(data[y_idx, x_idx])
        if save_opts is not None:
            try:
                assert isinstance(save_opts, dict)
                if save_opts["save"]:
                    f_name = save_opts["f_name"]
                    DataFrame(d).to_csv(f_name, index=False)
            except AssertionError:
                print(
                    "Please give the acceptable `save_opts` option! \n"
                    "type: `dict`, formation: {'save': 0 or 1, 'f_name': str, filename}"
                )
        else:
            return DataFrame(d)

    def discretize_domain(self):
        """
        :return: x : numpy array, shape (N, ) 1D array of N X-coordinates (horizontal direction).
                 y : numpy array, shape (M, ) 1D array of M z-coordinates (vertical direction).
        """
        x_min, x_max, y_min, y_max = self.layer_domain
        x = np.arange(x_min, x_max + 20, 20.0)
        y = np.arange(y_min, y_max + 20, 20.0)
        return x, y

    def layer_rho_kriging(self, layer_label):
        """
        :param layer_label: str, which layer needs to do kriging.
        :return: x : numpy array, shape (N, ) 1D array of N X-coordinates (horizontal direction).
                 y : numpy array, shape (M, ) 1D array of M y-coordinates (vertical direction).
                 grid_array : numpy array, shape (M, N) (M, N) array of grid values, where M is number
                 of Y-coordinates and N is number of X-coordinates. The array entry corresponding to the lower-left
                 coordinates is at index [M, 0], so that the array is oriented as it would be in X-Z space.
        """
        rho_layer = self.extract_layer_rho(layer_label)
        val = rho_layer.values
        ind = val[:, 2] > 0
        val = val[ind]
        x, y = self.discretize_domain()
        var_model = gs.Exponential(dim=2, len_scale=self.delta * 2, var=0.5)
        ok2d = OrdinaryKriging(
            val[:, 1],
            val[:, 0],
            val[:, 2],
            variogram_model=var_model,
            verbose=False,
            enable_plotting=False,
        )
        val, sigma_sq = ok2d.execute("grid", x, y)
        return x, y, val, sigma_sq
