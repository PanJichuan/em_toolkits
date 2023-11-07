import numpy as np
import pygimli as pg
from pygimli import meshtools as mt
from gempy.core.grid_modules import section_utils
from matplotlib.patches import PathPatch


class Model2D:
    def __init__(self, domain, delta=None, layer=None, **domain_kwargs, ):
        self.x_min, self.x_max, self.z_min, self.z_max = domain
        if delta is None:
            delta = 20
        self.dx = delta
        self.dz = delta
        self.x_pts = np.arange(self.x_min, self.x_max + 1, self.dx)
        self.y_pts = np.arange(self.z_min, self.z_max + 1, self.dz)

        self.world = mt.createWorld([self.x_min, self.z_min],
                                    [self.x_max, self.z_max],
                                    **domain_kwargs)
        if layer is not None:
            if isinstance(layer, list):
                z0, z1 = layer
                self.layer = mt.createLine(start=[self.x_min, z0],
                                           end=[self.x_max, z1],
                                           nSegments=int((self.x_max - self.x_min) / self.dx))
            else:
                z = layer
                self.layer = mt.createLine(start=[self.x_min, z],
                                           end=[self.x_max, z],
                                           nSegments=int((self.x_max - self.x_min) / self.dx))
            self.layer_nodes = [[pg.x(n), pg.y(n)] for n in self.layer.nodes()]

            self.plc = self.world+self.layer
        else:
            self.plc = self.world

    def add_rect(self, start, end, **kwargs):
        return mt.createRectangle(start, end, **kwargs) + self.plc

    @property
    def bool_operator_on_1layer(self):
        ind = []
        x_ind = (self.x_pts == self.layer_nodes[:, 0])
        for i, do in enumerate(x_ind):
            if do:
                y_ind = (self.y_pts >= abs(self.layer_nodes[i][1]))
                ind.append(y_ind)
        return np.transpose(ind)


class GemPyModelToGimliPoly:
    def __init__(self, geo_model, section, ignored_surface, z_shift, save_poly, show_poly):
        self.geo_model = geo_model
        self.section = section
        self.ignored_surface = ignored_surface
        self.z_shift = z_shift
        self.save_poly = save_poly
        self.show_poly = show_poly

    @property
    def get_vertices_from_geo_model(self):
        vertices, colors, extent = section_utils.get_polygon_dictionary(
            geo_model=self.geo_model,
            section_name=self.section,
        )
        return vertices

    def create_poly_from_geo_model(self):
        # get_region_extent
        vertices = self.get_vertices_from_geo_model
        geometry = {form: PathPatch(path[0]) for form, path in vertices.items()}
        all_vertices = [i.get_verts() for i in geometry.values()]
        x_max = max([max(i[:, 0]) for i in all_vertices])
        x_min = min([min(i[:, 0]) for i in all_vertices])
        z_max = max([max(i[:, 1]) for i in all_vertices])
        z_min = min([min(i[:, 1]) for i in all_vertices])
        # Setting up vertical shift of the model
        if self.z_shift == 1:
            self.z_shift = z_max
            z_min -= z_max
            z_max = 0
        else:
            z_max -= self.z_shift
            z_min -= self.z_shift

        # Create the region world
        world = mt.createWorld((x_min, z_max), (x_max, z_min))
        unit_count = 0

        for name, unit in vertices.items():
            if name not in self.ignored_surface:
                unit_count += 1
                to_world = np.array([nde[0] for nde in unit[0].iter_segments()])
                to_world[:, 1] -= self.z_shift
                # Remove redundant end nodes
                if (to_world[0] == to_world[-1]).all():
                    to_world = np.delete(to_world, -1, axis=0)
                # Add surface nodes to pyGIMLI world as a new polygon
                world += mt.createPolygon(to_world, isClosed=True, marker=unit_count)

        if self.show_poly is True:
            pg.show(world)

        if self.save_poly is True:
            mt.exportPLC(world, self.save_poly)

        return world
