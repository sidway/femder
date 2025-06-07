import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d
import pandas as pd

# Author: S.C. (07/06/2025)
# This function was written to calculate the modal decay time for the identified peaks in the SPL/FRF simulations provided by FEMDER computations
# Original paper: A. Prato, F. Casassa, P. Di Torino, A. Schiavi, A modal approach
# for reverberation time measurements in non-diffuse sound
# field, in: Proceedings of the 23rd International Congress on
# Sound & Vibration, Athens, 2016.
# 
#
# Plot with treshold to be implemented
# Results yet to be validated with measurements by the author. But has been explored before in DOI: 10.3280/ria2-2024oa17606 (Andrea Cicero)


def calculate_mt60_from_frf(freq, magnitude_db, min_peak_height=None, min_peak_distance=10,
                           smoothing_window=5, plot_results=True):
    """
    Calculate MT60 (modal decay times) from frequency response function data.

    Parameters:
    -----------
    freq : array-like
        Frequency array [Hz]
    magnitude_db : array-like
        Magnitude in dB (SPL or other dB scale)
    min_peak_height : float, optional
        Minimum peak height in dB above surrounding valleys
    min_peak_distance : int, optional
        Minimum distance between peaks in samples
    smoothing_window : int, optional
        Window size for Savitzky-Golay smoothing (odd number)
    plot_results : bool, optional
        Whether to plot the results

    Returns:
    --------
    results : dict
        Dictionary containing peak frequencies, bandwidths, and MT60 values
    """

    # Ensure arrays are numpy arrays
    freq = np.array(freq)
    magnitude_db = np.array(magnitude_db)

    # Smooth the data to reduce noise
    if smoothing_window > 1 and len(magnitude_db) > smoothing_window:
        magnitude_smooth = savgol_filter(magnitude_db, smoothing_window, 2)
    else:
        magnitude_smooth = magnitude_db.copy()

    # Find peaks
    if min_peak_height is None:
        # Automatically determine minimum peak height based on data range
        min_peak_height = (np.max(magnitude_smooth) - np.min(magnitude_smooth)) * 0.1

    peaks, peak_properties = find_peaks(magnitude_smooth,
                                       height=np.mean(magnitude_smooth) + min_peak_height,
                                       distance=min_peak_distance)

    # Initialize results storage
    results = {
        'peak_frequencies': [],
        'peak_magnitudes': [],
        'bandwidths': [],
        'mt60_values': [],
        'left_3db_freqs': [],
        'right_3db_freqs': []
    }

    # Create interpolation function for more precise -3dB point finding
    interp_func = interp1d(freq, magnitude_smooth, kind='linear',
                          bounds_error=False, fill_value='extrapolate')

    for peak_idx in peaks:
        peak_freq = freq[peak_idx]
        peak_mag = magnitude_smooth[peak_idx]
        target_mag = peak_mag - 3.0  # -3dB point

        # Find frequency range around the peak to search for -3dB points
        # Look for the valleys on either side of the peak
        left_valley_idx = 0
        right_valley_idx = len(magnitude_smooth) - 1

        # Find left valley (minimum point to the left of peak)
        for i in range(peak_idx - 1, 0, -1):
            if i == 0 or (magnitude_smooth[i] < magnitude_smooth[i-1] and
                         magnitude_smooth[i] < magnitude_smooth[i+1]):
                left_valley_idx = i
                break

        # Find right valley (minimum point to the right of peak)
        for i in range(peak_idx + 1, len(magnitude_smooth) - 1):
            if i == len(magnitude_smooth) - 1 or (magnitude_smooth[i] < magnitude_smooth[i-1] and
                                                  magnitude_smooth[i] < magnitude_smooth[i+1]):
                right_valley_idx = i
                break

        # Search for -3dB points between valleys and peak
        left_3db_freq = None
        right_3db_freq = None

        # Left side: find where magnitude crosses target_mag
        for i in range(left_valley_idx, peak_idx):
            if (magnitude_smooth[i] <= target_mag and magnitude_smooth[i+1] >= target_mag):
                # Linear interpolation for more precise frequency
                alpha = (target_mag - magnitude_smooth[i]) / (magnitude_smooth[i+1] - magnitude_smooth[i])
                left_3db_freq = freq[i] + alpha * (freq[i+1] - freq[i])
                break

        # Right side: find where magnitude crosses target_mag
        for i in range(peak_idx, right_valley_idx):
            if (magnitude_smooth[i] >= target_mag and magnitude_smooth[i+1] <= target_mag):
                # Linear interpolation for more precise frequency
                alpha = (target_mag - magnitude_smooth[i]) / (magnitude_smooth[i+1] - magnitude_smooth[i])
                right_3db_freq = freq[i] + alpha * (freq[i+1] - freq[i])
                break

        # Calculate bandwidth and MT60 if both -3dB points found
        if left_3db_freq is not None and right_3db_freq is not None:
            bandwidth = right_3db_freq - left_3db_freq
            mt60 = 2.2 / bandwidth  # source: Prato et al.

            # Store results
            results['peak_frequencies'].append(peak_freq)
            results['peak_magnitudes'].append(peak_mag)
            results['bandwidths'].append(bandwidth)
            results['mt60_values'].append(mt60)
            results['left_3db_freqs'].append(left_3db_freq)
            results['right_3db_freqs'].append(right_3db_freq)

    # Convert to numpy arrays
    for key in results:
        results[key] = np.array(results[key])

    # Plot results if requested
    if plot_results and len(results['peak_frequencies']) > 0:
        plt.figure(figsize=(12, 8))

        # Main plot
        plt.subplot(2, 1, 1)
        plt.semilogx(freq, magnitude_db, 'b-', alpha=0.7, label='Original FRF')
        plt.semilogx(freq, magnitude_smooth, 'r-', linewidth=2, label='Smoothed FRF')

        # Mark peaks and -3dB points
        plt.semilogx(results['peak_frequencies'], results['peak_magnitudes'],
                    'ro', markersize=8, label='Peaks')
        plt.semilogx(results['left_3db_freqs'], results['peak_magnitudes'] - 3,
                    'g^', markersize=6, label='-3dB points (left)')
        plt.semilogx(results['right_3db_freqs'], results['peak_magnitudes'] - 3,
                    'g^', markersize=6, label='-3dB points (right)')

        # Draw bandwidth lines
        for i in range(len(results['peak_frequencies'])):
            plt.semilogx([results['left_3db_freqs'][i], results['right_3db_freqs'][i]],
                        [results['peak_magnitudes'][i] - 3, results['peak_magnitudes'][i] - 3],
                        'g--', alpha=0.7)

        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude [dB]')
        plt.title('FRF Analysis with Peak Detection and -3dB Points')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # MT60 results plot
        plt.subplot(2, 1, 2)
        plt.semilogx(results['peak_frequencies'], results['mt60_values'], 'bo-',
                    markersize=6, linewidth=2)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('MT60 [s]')
        plt.title('Modal Decay Times (MT60)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        plt.xticks([20,40,60,80,100,120,160,200],[20,40,60,80,100,120,160,200]);

    return results

