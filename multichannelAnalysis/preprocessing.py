import numpy as np
import matlab.engine
from scipy.io import loadmat
from scipy.stats import norm
from scipy.signal import butter, filtfilt, freqs



def flatten(lst):
    """Flatten a list."""  # Tomasi
    return [y for l in lst for y in flatten(l)] \
       if isinstance(lst, (list, np.ndarray)) else [lst]


class preProcessing():
    """ preProcessing is the class used to preprocess the data. It includes several methods, as filters for 50Hz and harmonics, highpass and bandstop filters, wavelet transform and wavelet normalization.
    """
    def __init__(self, x, fs):
        """ Class initializer. We pass the signal and we modify it through the methods of the class.
        Parameters:
            x, signal, can be one dimensional or two
            fs, sampling frequency
        """
        self.x = x
        self.fs = fs


    def filter_50Hz(self, forder=5):
        """
        Filter 50 Hz and harmonics using a Butterworth IIR, and the filtfilt to linearize the phase.
        Parameters:
            x, numpy array, one dimensional signal
            fs, float, sampling frequency
            forder, int, order of the filter
        Returns:
            x, filtered signal
        """
        nyq = self.fs / 2  # nyquist frequency
        cutoff = 50.  # we want to cut all the multiples of 50 Hz
        halfwidth = 1.5
        norm_bandstop = cutoff * np.arange(1, nyq / 50)  # here we generate 50 Hz and harmonics
        for f in norm_bandstop:
            b, a = butter(forder, [(f-halfwidth) / nyq, (f+halfwidth) / nyq] , 'bandstop', analog=False)
            self.x = filtfilt(b, a, self.x)
        return self.x


    def cwt_allrecords(self, scale):
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

        Ts = 1. / self.fs
        n, s = (self.x).shape
        coefs = np.zeros((n, s), dtype=complex)

        with matlab.engine.start_matlab() as eng:
            freq = eng.scal2frq(matlab.double([scale]), 'cmor1-1', Ts)
            if freq > self.f:
                raise ValueError("central frequency exceeds the Nyquist frequency")
            for idx, row in enumerate(self.x):
                row_mat = matlab.double(list(row))
                coefs[idx] = np.round(np.asarray(eng.cwt(row_mat, matlab.double([scale]), 'cmor1-1', Ts)), decimals=5)

        return coefs, freq


    def filter_highpass(self, cutoff, forder=5):
        """
        High pass filter, Butterworth IIR filter with linearization of the response
        Parameters:
            x, np.array, one dimensional signal
            cutoff, float, cutoff frequency
            fs, sampling frequency
            forder, filter order
        Returns:
            filtered signal
        """
        norm_cutoff = cutoff / (self.fs / 2)
        b, a = butter(forder, norm_cutoff, 'highpass', analog=False)
        self.x =  filtfilt(b, a, self.x)
        return self.x

    def diffmethod(self):
        """
        Differential method. Given a signal it returns
        2 * (1 - cos(omega)) * | X(exp[jomega])|**2
        Parameters:
            self
        Returns:
            freqs, the vector of frequency
            transf, the transformation we get through the differential method
        """
        xfft = 1. / x.size * np.fft.fft(x)
        freqs = np.fft.fftfreq(self.x.size, 1./self.fs)
        fnyq = self.fs / 2
        transf =  2 * (1 - np.cos(2 * np.pi * freqs)) * ((np.abs(xfft))**2)
        return freqs, transf

    def tkeo(self, CWTcoefs):
        """
        Compute the Teager-Kaiser Operator Energy. Here the input is the wavelet transform of the signal for a fixed scale.
        < T_f[n,m], T_f[n,m]^* > - 0.5 * < T_f[n-1,m], T_f[n+1,m]^* >  - 0.5 * < T_f[n+1,m], T_f[n-1,m]^* >
        Parameters:
            self,
            CWTcoefs, wavelet coefficients for a fixed scale
        Returns:
            T-K operator energy value
        """

        cwtcoefs = np.append(np.append(0, cwtcoefs), 0)
        return np.abs(cwtcoefs[1:-1])**2 - 0.5 * cwtcoefs[0:-1] * np.conj(cwtcoefs[1:]) - 0.5 * np.conj(cwtcoefs[:-1]) * cwtcoefs[1:]


    def H0zscore(self, cwtcoefs):
        """
        Normalization of the wavelet coefficients - more documentation is needed
        Parameters:
            self,
            cwtcoefs, wavelet coefficients for a fixed scale
        Returns:
            normalized wavelet coefficients
        """
        repart = np.real(cwtcoefs)
        impart = np.imag(cwtcoefs)
        firstre, thirdre = np.percentile(repart, [0.25, 0.75])
        firstim, thirdim = np.percentile(impart, [0.25, 0.75])
        iqrre = thirdre - firstre
        iqrim = thirdim - firstim
        minre = firstre - 1.5 * iqrre
        maxre = thirdre + 1.5 * iqrre
        minim = firstim - 1.5 * iqrim
        maxim = thirdim + 1.5 * iqrim

        distre = repart[np.logical_and(repart >= minre, repart <= maxre)]
        distim = impart[np.logical_and(impart >= minim, impart <= maxim)]

        # fit gaussian around these two values and extract the mean and stardard deviation
        mure, sigmare = norm.fit(distre)
        muim, sigmaim = norm.fit(distim)

        # standardize
        repart = (repart - mure) / sigmare
        impart = (impart- muim) / sigmaim

        return repart + 1j*impart
