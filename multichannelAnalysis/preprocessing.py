import numpy as np
import matlab.engine
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, freqs



def flatten(lst):
    """Flatten a list."""  # Tomasi
    return [y for l in lst for y in flatten(l)] \
       if isinstance(lst, (list, np.ndarray)) else [lst]


class preProcessing():
    """ preProcessing is the class used to preprocess the data. It includes several methods, as filters for 50Hz and harmonics, highpass and bandstop filters, wavelet transform and wavelet normalization.
    """
    def __init__(self, x):
        """ Class initializer. We pass the signal and we modify it through the methods of the class.
        Parameters:
            x, signal, can be one dimensional or two
        """
        self.x = x


    def filter_50Hz(self, fs=1000., forder=5):
        """
        Filter 50 Hz and harmonics using a Butterworth IIR, and the filtfilt to linearize the phase.
        Parameters:
            x, numpy array, one dimensional signal
            fs, float, sampling frequency
            forder, int, order of the filter
        Returns:
            x, filtered signal
        """
        nyq = fs / 2  # nyquist frequency
        cutoff = 50.  # we want to cut all the multiples of 50 Hz
        halfwidth = 1.5
        norm_bandstop = cutoff * np.arange(1, nyq / 50)  # here we generate 50 Hz and harmonics
        for f in norm_bandstop:
            b, a = butter(forder, [(f-halfwidth) / nyq, (f+halfwidth) / nyq] , 'bandstop', analog=False)
            self.x = filtfilt(b, a, self.x)
        return self.x


    def cwt_allrecords(self, scale, Ts=1e-3):
        """
        Given the matrix of all recordings this function compute the wavelet transform for a specific scale.
        Parameters:
            x, np.array of dimension (#recordings, #time)
            scale_id, identifier for the scale, this index goes from 0 to 100
            Ts, sampling period, default 1 ms
        Returns:
            coefs, np.array of complex number containing the coefficients of the wavelet transform. We use the Morlet as mother wavelet
            freq, the central frequency of the wavelet
        """
        # mins, maxs, ns = 0.3, 3, 100
        # scales = np.logspace(mins, maxs, ns)
        # scale = scales[scale_id]

        n, s = (self.x).shape
        coefs = np.zeros((n, s), dtype=complex)

        with matlab.engine.start_matlab() as eng:
            freq = eng.scal2frq(matlab.double([scale]), 'cmor1-1', Ts)
            for idx, row in enumerate(self.x):
                row_mat = matlab.double(list(row))
                coefs[idx] = np.round(np.asarray(eng.cwt(row_mat, matlab.double([scale]), 'cmor1-1', Ts)), decimals=5)

        return coefs, freq


    def filter_highpass(self, cutoff, fs=1000., forder=5):
        """ High pass filter, Butterworth IIR filter with linearization of the response
        Parameters:
            x, np.array, one dimensional signal
            cutoff, float, cutoff frequency
            fs, sampling frequency
            forder, filter order
        Returns:
            filtered signal
            """
        norm_cutoff = cutoff / (fs / 2)
        b, a = butter(forder, norm_cutoff, 'highpass', analog=False)
        self.x =  filtfilt(b, a, self.x)
        return self.x
