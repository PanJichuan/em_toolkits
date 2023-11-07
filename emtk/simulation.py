import empymod
import matplotlib.pyplot as plt
import numpy as np
import pygimli as pg
from pygimli import meshtools as mt
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline
from scipy.special import roots_legendre


def get_time(off_time, waveform_time):
    """Additional time for ramp.
    Parameters
    ----------
    off_time : `np.array`
        Time channels'
    waveform_time : `np.array`
        Waveform times
    Returns
    -------
    time_req : `np.array`
        Required times: Range from the min waveform time to the max off time
    """
    t_min = np.log10(max(off_time.min() - waveform_time.max(), 1e-10))
    t_max = np.log10(off_time.max() - waveform_time.min())
    return np.logspace(t_min, t_max, off_time.size + 2)


def waveform(times, resp, times_wanted, wave_time, wave_amp, n_quad=3):
    """Apply a source waveform to the signal.
    Parameters
    ----------
    times : `np.array`
        Times of computed input response; should start before and end after
        `times_wanted`.
    resp : `np.array`
        EM-response corresponding to `times`.
    times_wanted : `np.array`
        Wanted times.
    wave_time : `np.array`
        Time steps of the wave.
    wave_amp : `np.array`
        Amplitudes of the wave corresponding to `wave_time`, usually
        in the range of [0, 1].
    n_quad : int
        Number of Gauss-Legendre points for the integration. Default is 3.
    Returns
    -------
    resp_wanted : `np.array`
        EM field for `times_wanted`.
    """
    # Interpolate on log.
    spl = iuSpline(np.log10(times), resp)
    # Wave time steps.
    dt = np.diff(wave_time)
    dI = np.diff(wave_amp)
    dIdt = dI / dt
    # Gauss-Legendre Quadrature; 3 is generally good enough.
    # (Roots/weights could be cached.)
    g_x, g_w = roots_legendre(n_quad)
    # Pre-allocate output.
    resp_wanted = np.zeros_like(times_wanted)
    # Loop over wave segments.
    for i, cdIdt in enumerate(dIdt):
        # We only have to consider segments with a change of current.
        if cdIdt == 0.0:
            continue
        # If wanted time is before a wave element, ignore it.
        ind_a = wave_time[i] < times_wanted
        if np.sum(ind_a) == 0:
            continue
        # If wanted time is within a wave element, we cut the element.
        ind_b = wave_time[i + 1] > times_wanted[ind_a]
        # Start and end for this wave-segment for all times.
        ta = times_wanted[ind_a] - wave_time[i]
        tb = times_wanted[ind_a] - wave_time[i + 1]
        tb[ind_b] = 0.0  # Cut elements
        # Gauss-Legendre for this wave segment. See
        # https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval
        # for the change of interval, which makes this a bit more complex.
        log_t = np.log10(np.outer((tb - ta) / 2, g_x) + (ta + tb)[:, None] / 2)
        fact = (tb - ta) / 2 * cdIdt
        resp_wanted[ind_a] += fact * np.sum(np.array(spl(log_t) * g_w), axis=1)
    return resp_wanted


class TDEMSimulation:
    def __init__(self, moment, source, receiver, model, depth, arbitrary_receiver=False):
        """
        Parameters
        ----------
        moment : str {'lm', 'hm'}
            Moment. If 'lm', above defined ``lm_off_time``, ``lm_waveform_times``,
            and ``lm_waveform_current`` are used. Else, the corresponding
            ``hm_``-parameters.
        source : list of floats or arrays
            [x0, x1, y0, y1, z0, z1]. El. bipole source; half of one side, See 'empymod.model.bipole' for more details.
            If the receiver wouldn't be in the center, we would have to model
            the actual complete loop (no symmetry to take advantage of).
            Then source coordinates has to be four points locations.
            Examples:
                source=[[20, 20, -20, -20],  # x1
                        [20, -20, -20, 20],  # x2
                        [-20, 20, 20, -20],  # y1
                        [20, 20, -20, -20],  # y2
                        0, 0],               # z1, z2
        receiver : list of floats or arrays
            [x, y, z, azimuth, dip]. See 'empymod.model.bipole' for more details.
        model : ndarray or list
            Resistivity of the resistivity model (see ``empymod.model.bipole``
            for more info.)
        depth : ndarray or list
            Absolute depths of the resistivity model, starting from topography and
            having the same length with model (len(depth)=len(model)). For example,
            if the surface elevation is 0m, then `depth` will be look like this:
            [0, z1, z2, ..., z_n] where z_i > 0.
            (see ``empymod.model.bipole`` for more info.)
        arbitrary_receiver: bool
            Whether the receiver location is in the center of the square loop.

        """
        self.moment = moment
        # Low moment off time
        if self.moment == 'lm':
            self.off_time = np.array([
                1.149E-05, 1.350E-05, 1.549E-05, 1.750E-05, 2.000E-05, 2.299E-05,
                2.649E-05, 3.099E-05, 3.700E-05, 4.450E-05, 5.350E-05, 6.499E-05,
                7.949E-05, 9.799E-05, 1.215E-04, 1.505E-04, 1.875E-04, 2.340E-04,
                2.920E-04, 3.655E-04, 4.580E-04, 5.745E-04, 7.210E-04])
            self.waveform_times = np.r_[-1.041E-03, -
                                        9.850E-04, 0.000E+00, 4.000E-06]
            self.waveform_current = np.r_[0.0, 1.0, 1.0, 0.0]
            self.waveform_label = 'Low moment'
        # High moment off time
        elif self.moment == 'hm':
            self.off_time = np.array([
                9.810e-05, 1.216e-04, 1.506e-04, 1.876e-04, 2.341e-04, 2.921e-04,
                3.656e-04, 4.581e-04, 5.746e-04, 7.211e-04, 9.056e-04, 1.138e-03,
                1.431e-03, 1.799e-03, 2.262e-03, 2.846e-03, 3.580e-03, 4.505e-03,
                5.670e-03, 7.135e-03])
            self.waveform_times = np.r_[-8.333E-03, -
                                        8.033E-03, 0.000E+00, 5.600E-06]
            self.waveform_current = np.r_[0.0, 1.0, 1.0, 0.0]
            self.waveform_label = 'High moment'
        else:
            raise ValueError("Moment must be either 'lm' or 'hm'!")
        # === Get required times, start after transmitter off, end before off time over
        self.require_time = get_time(self.off_time, self.waveform_times)
        # Transmitter signal type
        self._signal = -1   # Switch-off response
        self.delay_rst = 1.8e-7  # As stated in the WalkTEM manual
        # === GET REQUIRED FREQUENCIES ===
        self.time, self.freq, self.ft, self.f_targ = empymod.utils.check_time(
            time=self.require_time,  # Required times
            signal=self._signal,  # Switch-on response
            ft='dlf',  # Use DLF
            ftarg={'dlf': 'key_81_CosSin_2009'},  # Short, fast filter; if you
            verb=1,  # need higher accuracy choose a longer filter.
        )

        self.source = source
        self.receiver = receiver
        self.params = {
            'src': self.source,
            'rec': self.receiver,
            'depth': depth,
            'res': np.r_[2e14, model],
            'freqtime': self.freq,  # Required frequencies.
            'mrec': True,  # It is an el. source, but a magnetic. rec.
            'strength': 8,  # To account for 4 sides of square loop.
            'srcpts': 3,  # Approx. the finite dip. with 3 points.
            'htarg': {'dlf': 'key_101_2009'}  # Short filter, so fast.
        }
        self.arbitrary_rec = arbitrary_receiver
        if self.arbitrary_rec:
            self.params['strength'] = 1
        else:
            pass

    def show_waveform(self):
        ax = plt.figure().add_subplot()
        ax.plot(np.r_[-9, self.waveform_times * 1e3, 2],
                np.r_[0, self.waveform_current, 0],
                label=self.waveform_label)
        ax.set(xlabel='Time (ms)', ylabel='Current (A)', xlim=(-9, 0.5))
        plt.show()

    @property
    def frequency_domain_response(self, verbose=1):
        """
        Parameters
        ----------
        verbose: int
        Returns
        -------
        FEM response: EMArray
            FEM response
        """
        # === COMPUTE FREQUENCY-DOMAIN RESPONSE ===
        response = empymod.model.bipole(**self.params, verb=verbose)
        if self.arbitrary_rec:
            return response.sum(axis=1)  # Sum all source bipoles
        else:
            return response

    @property
    def time_domain_response(self):
        f_em = self.frequency_domain_response

        # Multiply the frequency-domain result with
        # \mu for H->B, and i\omega for B->dB/dt.
        f_em *= 2j * np.pi * self.freq * 4e-7 * np.pi
        cut_off_freq = 4.5e5  # As stated in the WalkTEM manual
        h = (1 + 1j * self.freq / cut_off_freq) ** -1  # First order type
        h *= (1 + 1j * self.freq / 3e5) ** -1
        f_em *= h
        # === CONVERT TO TIME DOMAIN ===
        # delay_rst = 1.8e-7  # As stated in the WalkTEM manual
        tem, _ = empymod.model.tem(fEM=f_em[:, None],  # fEM
                                   off=np.array([1]),  # Off
                                   freq=self.freq,  # freq
                                   time=self.time + self.delay_rst,  # time
                                   signal=self._signal,  # signal
                                   ft=self.ft,
                                   ftarg=self.f_targ)
        tem = np.squeeze(tem)
        return tem

    @property
    def db_dt(self):
        tem = self.time_domain_response
        return waveform(self.time,
                        tem,
                        self.off_time,
                        self.waveform_times,
                        self.waveform_current)

    def synthetic_data(self, error_level: float):
        data = self.db_dt
        error = np.ones_like(data) * error_level
        return data * (np.random.randn(len(data)) * error + 1.0), error


class TEMForward(pg.Modelling):
    def __init__(self, moment, source, receiver, depth: list):
        """
        Parameters
        ----------
        moment : str {'lm', 'hm'}
            Moment. If 'lm', above defined ``lm_off_time``, ``lm_waveform_times``,
            and ``lm_waveform_current`` are used. Else, the corresponding
            ``hm_``-parameters.
        source : list of floats or arrays
            [x0, x1, y0, y1, z0, z1]. See 'empymod.model.bipole' for more details.
        receiver : list of floats or arrays
            [x, y, z, azimuth, dip]. See 'empymod.model.bipole' for more details.
        depth : list
            Depths of the resistivity model (see ``class:: TDEMSimulation`` for more
            info.)
        """
        self._moment = moment
        self._src = source
        self._rec = receiver
        self._dep = depth
        self.mesh1d = mt.createMesh1D(len(self._dep))
        super().__init__()
        self.setMesh(self.mesh1d)

    def response(self, model):
        """
        Parameters
        ----------
        model : ndarray
            Resistivity of the resistivity model (see ``empymod.model.bipole``
            for more info.)
        Returns
        -------
            Time domain em field (ndarray), $\frac{dB}{dt}$
        """
        fop = TDEMSimulation(self._moment, self._src,
                             self._rec, model, self._dep)
        return fop.db_dt

    def createStartModel(self, **dataVal):
        return pg.Vector(len(self.dep)) * 100
