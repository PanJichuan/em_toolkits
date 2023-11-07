import pygimli as pg
from .simulation import TEMForward


class TDEMInversion:
    def __init__(self, moment, source, receiver, depth, data, error, verb=False, reg_params=None):
        """
        Parameters
        ----------
        moment : str {'lm', 'hm'}
            TEM instrument work terms, set each off time, waveform time and waveform
            current params.
        source ： list of floats or arrays
            [x0, x1, y0, y1, z0, z1]. See 'empymod.model.bipole' for more details.
        receiver : list of floats or arrays
            [x, y, z, azimuth, dip]. See 'empymod.model.bipole' for more details.
        depth : ndarray or list
            Smooth depth, usually set by `np.linspace(z_min, z_max, n_nodes)`.
        data : ndarray or list
            Observed em field data.
        error : ndarray or list
            The error level of each datum.
        verb : bool
            Whether to show the inversion procedure info.
        reg_params : kwargs
            The regularization params, see 'pygimli.Inversion.setRegularization'
            for more info.
        """
        self._moment = moment
        self._src = source
        self._rec = receiver
        self._dep = depth
        self._data = data
        self._error = error
        self._verb = verb
        if reg_params is not None:
            self._reg_params = reg_params
        else:
            self._reg_params = None
        self._fop_params = {
            'moment': self._moment,
            'source': self._src,
            'receiver': self._rec,
            'depth': self._dep,
        }

    def go(self, **kwargs):
        """
        Parameters
        ----------
        kwargs: dict
            Parameters of `pg.Inversion().run()`
        Returns
        -------
            `class:: pg.Inversion()` and recovered model.
        """
        fop = TEMForward(**self._fop_params)
        inv = pg.Inversion()
        inv.setForwardOperator(fop)
        transModel = pg.trans.TransLog(1)
        inv.transModel = transModel
        if self._reg_params is not None:
            inv.setRegularization(**self._reg_params)
        model = inv.run(self._data, self._error, verbose=self._verb, **kwargs)
        return inv, model
