# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 09:46:26 2023

@author: kemer https://github.com/biancakemerich
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
import scipy.signal.windows as win
import scipy.signal as signal
import more_itertools
import femder as fd
import pytta 
from matplotlib import gridspec

def fft(time_data, axis=0, crop=True):
    """
    Compute the FFT of a time signal.

    Parameters
    ----------
    time_data : array
        Time data array.
    axis : int, optional
        NumPy's axis option.
    crop : bool, optional
        Option the remove the second half of the FFT.

    Returns
    -------
    Frequency response array.
    """

    freq_data = 1 / len(time_data) * np.fft.fft(time_data, axis=axis)

    if crop:
        freq_data = freq_data[0:int(len(freq_data) / 2) + 1]

    return freq_data

def pressure2spl(pressure, ref=2e-5):
    """
    Computes Sound Pressure Level [dB] from complex pressure values [Pa].

    Parameters
    ----------
    pressure: array
        Complex pressure array.

    Returns
    -------
    SPL array.
    """
    spl = 10 * np.log10(0.5 * pressure * np.conj(pressure) / ref ** 2)

    return np.real(spl)

def impulse_response_calculation(freq, ir, values_at_freq, freq_filter_indices, tukey, alpha, rolling_window,
                                 cut_sample, filter_sample, n, return_id, high_pass_values, low_pass_values,
                                 base_fontsize=15, linewidth=2, figsize=(16, 20)):
    """
    Auxiliary plot to view impulse response calculation results, windows and filters.

    Parameters
    ----------
    freq : array
        Frequency vector.
    ir : list
        List containing calculated IRs.
    values_at_freq : array
        Complex pressure values at the receiver.
    freq_filter_indices : array
        Indices of the frequency range of analysis.
    tukey : array
        Tukey window used to filter the noise at the end of the IR during post-processing.
    alpha : float
        Alpha value of the pre-processing Tukey window.
    rolling_window : iterable
        List of rolling windows.
    cut_sample : int
        Sample of the window of lowest energy.
    filter_sample : int
        Sample at which the Tukey window will start to remove the noise at the end.
    n : int
        Size of the rolling windows in samples.
    return_id : int
        Index of which IR from the IR list will be returned.
    high_pass_values : array
        High-pass filter frequency response.
    low_pass_values : array
        Low-pass filter frequency response.
    base_fontsize : int
        Base font size.
    linewidth : int
        Plot line width.
    figsize : tuple
        Matplotlib figure size.

    Returns
    -------
    Matplotlib figure, Gridspec object and list of Matplotlib axes.
    """

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 1)
    ax = [plt.subplot(gs[i, 0]) for i in range(4)]

    i = 0

    ax[i].set_title(f"Impulse Response", size=base_fontsize + 2, fontweight="bold", loc="left")
    ax[i].set_xlabel("Samples [n]", size=base_fontsize)
    ax[i].set_ylabel("Amplitude [-]", size=base_fontsize)
    ax[i].plot(ir[0], alpha=0.3, linewidth=linewidth, label="Filtered Input Data")
    ax[i].axvline(filter_sample * n, color="b", label="Filter start point")
    ax[i].plot(tukey * max(ir[0]), label="Filter window", linewidth=linewidth)
    for r in range(len(rolling_window) + 1):
        ax[i].axvline(r * n, color="k", zorder=0, alpha=0.3, linewidth=linewidth, linestyle=":",
                      label="Rolling windows" if r == 0 else None)
    ax[i].axvline(cut_sample * n, color="r", label="Lowest energy", linewidth=linewidth, linestyle="--")
    ax[i].plot(ir[return_id], alpha=0.7, linewidth=linewidth, label="Processed IR")
    i += 1

    ax[i].set_title(f"Frequency Response", size=base_fontsize + 2, fontweight="bold", loc="left")
    ax[i].set_xlabel("Frequency [Hz]", size=base_fontsize)
    ax[i].set_ylabel("Amplitude [dB]", size=base_fontsize)
    ax[i].plot(freq, pressure2spl(np.abs(values_at_freq)), label="Input Data", linewidth=linewidth)
    ax[i].plot(freq, pressure2spl(np.abs(fft(ir[return_id])[freq_filter_indices])),
               label="Processed IR", linewidth=linewidth, linestyle=":")
    i += 1

    ax[i].set_title(f"Phase Response", size=base_fontsize + 2, fontweight="bold", loc="left")
    ax[i].set_xlabel("Frequency [Hz]", size=base_fontsize)
    ax[i].set_ylabel("Angle [deg]", size=base_fontsize)
    ax[i].plot(freq, np.rad2deg(np.angle(values_at_freq)), label="Input Data", linewidth=linewidth)
    ax[i].plot(freq, np.rad2deg(np.angle(fft(ir[return_id])[freq_filter_indices])),
               label="Processed IR",
               linewidth=linewidth)
    i += 1


    ax[i].set_title(f"Filters and windows response", size=base_fontsize + 2, fontweight="bold", loc="left")
    ax[i].set_xlabel("Frequency [Hz]", size=base_fontsize)
    ax[i].set_ylabel("Amplitude [dBFS]", size=base_fontsize)
    ax[i].plot(freq, pressure2spl(abs(high_pass_values), ref=1), label="High-pass",
               linewidth=linewidth)
    ax[i].plot(freq, pressure2spl(abs(low_pass_values), ref=1), label="Low-Pass",
               linewidth=linewidth)
    ax[i].plot(freq, pressure2spl(abs(signal.tukey(len(freq), alpha)), ref=1),
               label="Tukey", linewidth=linewidth)
    i += 1

    for j in range(i):
        ax[j].legend(fontsize=base_fontsize - 2, loc="best", ncol=6)
        ax[j].grid("minor")
        ax[j].tick_params(axis='both', which='both', labelsize=base_fontsize - 3)

    gs.tight_layout(fig, pad=1)

    return fig, gs, ax
    

class Domain:
    """Acoustical domain properties and methods."""
    def __init__(self, fmin, fmax, tmax, fs=44100):
        """
        Parameters
        ----------
        fmin : float
            Minimum sampling frequency.
        fmax : float
            Maximum sampling frequency.
        tmax : float
            Time in seconds until which to sample the room impulse response.
        fs : int, optional
            Sampling rate in Hz for the time signal.
        """
        self._fmin = fmin
        self._fmax = fmax
        self._tmax = tmax
        self._fs = fs
        self._high_pass_freq = 2 * self.fmin #self._high_pass_freq = 2 * self.fmin
        self._low_pass_freq = 2 * self.fmax #self._low_pass_freq = 2 * self.fmax
        self._high_pass_order = 4
        self._low_pass_order = 4
        self._freq_overhead = [2, 1.2]
        self._alpha = None
        self._air_prop = fd.AirProperties()

    @property
    def air_prop(self):
        """Return air properties dictionary."""
        return self._air_prop.standardized_c0_rho0()

    @property
    def num_freq(self):
        """Return number of frequencies."""
        return int(round(self.fs * self.tmax))

    @property
    def fs(self):
        """Return sampling rate."""
        return self._fs

    @property
    def tmax(self):
        """Return time duration."""
        return self._tmax

    @property
    def time(self):
        """Return time steps."""
        return np.arange(self.num_freq, dtype="float64") / self.fs

    @property
    def all_freqs(self):
        """Return frequencies."""
        return self.fs * np.arange(self.num_freq, dtype="float64") / self.num_freq

    @property
    def df(self):
        """Return frequency resolution."""
        return self.fs / self.num_freq

    # @property
    # def freq_overhead(self):
    #     """Return the lower and upper overhead range factors."""
    #     return self._freq_overhead

    @property
    def freq_filter_indices(self):
        """Return the indices of the filtered frequencies."""
        # return np.flatnonzero((self.all_freqs >= self.fmin / self.freq_overhead[0])
                               # & (self.all_freqs <= self.fmax * self.freq_overhead[1]))

        # return np.flatnonzero(((self.all_freqs >= self.fmin)
        #                        & (self.all_freqs <= self.fmax)))

        return np.flatnonzero(((self.all_freqs >= self.fmin) & (self.all_freqs <= self.fmax)))

    @property
    def freq(self):
        """Return the filtered frequencies."""
        return self.all_freqs[self.freq_filter_indices]

    @property
    def w0(self):
        """Return the filtered frequencies."""
        return 2 * np.pi * self.freq

    @property
    def fmin(self):
        """Return minimum frequency."""
        return self._fmin

    @property
    def fmax(self):
        """Return maximum frequency."""
        return self._fmax

    @property
    def df(self):
        """Return frequency resolution."""
        return 1 / self.tmax

    @property
    def high_pass_freq(self):
        """Return high pass frequency."""
        return self._high_pass_freq

    @high_pass_freq.setter
    def high_pass_freq(self, freq):
        """Set high pass frequency."""
        self._high_pass_freq = freq

    @property
    def low_pass_freq(self):
        """Return low pass frequency."""
        return self._low_pass_freq

    @low_pass_freq.setter
    def low_pass_freq(self, freq):
        """Set low pass frequency."""
        self._low_pass_freq = freq

    @property
    def high_pass_filter_order(self):
        """Return high pass filter order."""
        return self._high_pass_order

    @high_pass_filter_order.setter
    def high_pass_filter_order(self, order):
        """Set high pass filter order."""
        self._high_pass_order = order

    @property
    def low_pass_filter_order(self):
        """Return low pass filter order."""
        return self._low_pass_order

    @low_pass_filter_order.setter
    def low_pass_filter_order(self, order):
        """Set low pass filter order."""
        self._low_pass_order = order

    @property
    def alpha(self):
        """Return Tukey window alpha value."""
        return self._alpha

    @alpha.setter
    def alpha(self, alphaValue):
        """Set Tukey window alpha value."""
        self._alpha = alphaValue
        
    def bands(self, n_oct=3):
        """Return octave frequency bands."""
        # bands = pytta.utils.freq.fractional_octave_frequencies(nthOct=n_oct)  # [band_min, band_center, band_max]
        return pytta.utils.freq.fractional_octave_frequencies(freqRange=(self.fmin, self.fmax), nthOct=n_oct)[:, 1]
        # return bands[np.argwhere((bands[:, 1] >= min(self.freq)) & (bands[:, 1] <= max(self.freq)))[:, 0]]

    def compute_impulse_response(self, values_at_freq, alpha=None, auto_roll=False, auto_window=True, irr_filters=False,
                                 view=False):
        """
        Compute the room impulse response.

        Parameters
        ----------
        values_at_freq : array
            The frequency domain values to be transformed taken at the filtered frequencies.
        alpha : float or None
            Tukey window alpha value. If 'None' is passed it will be automatically calculated.
        auto_rool : bool, optional
            Option to automatically roll the impulse response to ensure no noise at the end.
        auto_window : bool, optional
            Option to automatically filter the impulse response to ensure no noise at the end.
        irr_filters : bool, optionak
            Option to use additional low and high-pass filters during the preconditioning of the input data.
        view : bool, optional
            Option to show plot with different parameters for analysis.

        Returns
        ------
        An array of approximate time values at the given time steps.
        """

        # Applying low and high-pass filters
        """
        This minimizes the phase differences from the input pressure data to the FFT of the  computed impulse response.
        Filters of order higher than 4 are unstable in Scipy, consider replacing this by FIR filters in the future.
        """
        # Retorna os coeficientes do filtro butter p altas 
        # Retorna os coeficiente do filtro butter p baixas
        b_high, a_high = signal.butter(self.high_pass_filter_order, self.high_pass_freq*2 / self.fs, "high")
        b_low, a_low = signal.butter(self.low_pass_filter_order, self.low_pass_freq*2 / self.fs, "low")
        # Computa a resposta em frequencia do filtro digital
        _, high_pass_values = signal.freqz(b_high, a_high, self.freq, fs=self.fs)
        _, low_pass_values = signal.freqz(b_low, a_low, self.freq, fs=self.fs)
        butter_filtered_values = (values_at_freq * np.conj(low_pass_values) * np.conj(high_pass_values))

        # Applying Tukey window
        self.alpha = max([self.fmax - self.low_pass_freq, self.high_pass_freq - self.fmin]) / (
                    self.fmax - self.fmin) if alpha is None else alpha
        windowed_values = butter_filtered_values * signal.tukey(len(self.freq),
                                                                self.alpha) if irr_filters \
            else values_at_freq * signal.tukey(len(self.freq), self.alpha)
        full_freq_values = np.zeros(self.num_freq, dtype="complex64")
        full_freq_values[self.freq_filter_indices] = windowed_values
        full_freq_values[-self.freq_filter_indices] = np.conj(windowed_values)

        # Array containing the IR at different calculation points - [raw, rolled, filtered]
        ir = [np.real(np.fft.ifft(full_freq_values) * self.num_freq) for _ in range(3)]
        return_id = 0

        # Identifying noise at the end
        parts = 100
        n = int(self.fs / parts)  # Dividing the impulse response into equal parts
        rolling_window = list(more_itertools.windowed(ir[0], step=n, n=n, fillvalue=0))  # Creating the rolling window
        delta_amp = [np.sum(np.abs(win) ** 2) for win in rolling_window]  # Getting the total energy of each slice

        # Rolling the ir to remove move noise at the end
        cut_sample = delta_amp.index(np.min(delta_amp)) + 1  # Finding the slice with the smallest amplitude variation
        cut_size = (len(rolling_window) - cut_sample) * n
        ir[1] = np.roll(ir[1], cut_size)  # Rolling the impulse response

        # Filter the IR to remove noise at the end
        filter_sample = delta_amp.index(np.min(delta_amp)) - int(
            parts / (parts * 0.75))  # Finding the slice with the smallest amplitude variation
        filter_size = (len(rolling_window) - filter_sample) * n
        tukey = np.concatenate((np.ones(filter_sample * n), signal.tukey(filter_size * 2, 1)[filter_size::]))
        ir[2] = ir[2] * tukey

        if auto_roll:
            return_id = 1
        if auto_window:
            return_id = 2

        # Plotting
        if view:
            _, _, _ = impulse_response_calculation(self.freq, ir, values_at_freq, self.freq_filter_indices, tukey,
                                                        self.alpha, rolling_window, cut_sample, filter_sample, n,
                                                        return_id, high_pass_values, low_pass_values)

        return ir[return_id]
