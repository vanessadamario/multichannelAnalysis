from scipy.signal import iirfilter, lfilter
from sklearn.preprocessing import scale
import numpy as np
import pywt


# flatten a list
def flatten(x):
    return [y for l in x for y in flatten(l)] \
        if type(x) in (list, np.ndarray) else [x]

# filter_
def bandfilter(X, fs, centralf, band, order=2, filtertype='butter', filteraction='bandpass'):
    """
    Filter the time series through bandpass/ bandstop filter
    -------------
    Parameters
        X, matrix of signals, #recordings times #nsamples
        fs, sampling frequency
        centralf, central frequency of the filter
        band, band width of the filter
        order, order of the filter, default 2
        filtertype, default Butterworth
        filteraction, 'bandpass' or 'bandstop'
    -------------
    Returns
        Filtered signal
    """
    nyq = fs / 2.
    low = centralf - band / 2.  # symmetric with respect to the central frequency
    high = centralf + band / 2.
    low = low / nyq  # normalized with respect to the Nyquist frequency
    high = high / nyq
    b, a = iirfilter(order, [low, high], btype=filteraction,
                     analog=False, ftype=filtertype)  # iir filter

    return lfilter(b, a, X, axis=1)

# notch
def remove_powerline(X, fs, powerline=50., bandstopwidth=2., filterorder=2):
    """
    Filter 50 Hz and all the harmonics
    It calls band_filter function
    -------------
    Parameters
        X, matrix of signals, #recordings times #nsamples
        fs, sampling frequency
        powerline, 50Hz if Europe, 60Hz for USA
        bandstopwidth, width of the bandstop filter, default 2 Hz
        filterorder, order of the filter, default 2
    -------------
    Returns
        Filtered signal
    """
    nyq = fs / 2.  # nyquist frequency
    harmonics = np.linspace(powerline, nyq - powerline, nyq / powerline - 1)  # not over nyq
    filtertype = 'butter'
    filteraction = 'bandstop'
    for f in harmonics:
        X = bandfilter(X, fs, f, bandstopwidth)
    return X



class featureExtractionTimeSeries():
    """
    This is the class used to compute the features from time series.
    It is address to the analysis of neural recordings (SEEG signals).
    It computes basics features related to the spectrum of the distribution.
    First it removes the powerline effects. Its methods compute
    (i) the first moments of the distributions, from 1 to 4,
    (ii) Fourier transform features for several bands
    (iii) Shannon entropy
    (iv) time over a threshold
    """
    def __init__(self, X, fs, rmpowerline=True, powerline=50.):
        """
        Class initializer
        -------------
        Parameters
            X, matrix of signals, #recordings times #nsamples
            fs, sampling frequency
            powerline, 50Hz if Europe, 60Hz for USA
        -------------
        Attributes
            n, #recordings
            p, #time samples
        """
        self.X = X
        self.n, self.p = X.shape
        self.fs = fs
        self.rmpowerline = rmpowerline
        self.powerline = powerline

        if self.rmpowerline:
            self.X = remove_powerline(self.X, self.fs, self.powerline)


    # moments
    def computemoments(self):
        """
        First moments of the distribution
        -------------
        Parameters
            self
        -------------
        Returns
            moments, dimensions #recordings x moments [1,2,3,4]
        """
        moments = np.array([]).reshape(self.n, 0)

        skewness = ((self.X - self.X.mean(axis=1).reshape(-1, 1))**3).mean(axis=1) / (self.X.std(axis=1)**3)
        kurtosis = ((self.X - self.X.mean(axis=1).reshape(-1, 1))**4).mean(axis=1) / (self.X.std(axis=1)**4)

        # moments = np.hstack((moments, self.X.mean(axis=1).reshape(self.n, 1)))
        moments = np.hstack((moments, self.X.std(axis=1).reshape(self.n, 1)))
        moments = np.hstack((moments, skewness.reshape(self.n, 1)))
        moments = np.hstack((moments, kurtosis.reshape(self.n, 1)))

        return moments


    def computefftfeat(self):
        """
        It computes the features relative to the spectrum of the signal.
        The features are normalized with respect to the total energy at all
        frequencies, for each recording.
        This allows us to make a comparison between recordings.
        -------------
        Parameters
            self
        -------------
        Returns
            featurematrix, a matrix which has #recordings rows and
            a number of features related to the number of bands
        """
        absXfft = np.abs(np.fft.fft(self.X, axis=1))[:, 0:self.p//2]
        freqs = np.fft.fftfreq(self.p, (1. / self.fs))[0:self.p//2]  # [0, Nyq]

        # cumulative sum over all bands
        cumulative = np.cumsum(absXfft, axis=1)

        # list of physiological frequencies
        # alpha - delta - theta - beta - gamma
        physiological_bands = np.array([0., 1., 4., 8., 13., 30., 70.])
        # higher bands - from high gamma to Nyquist
        pathological_bands = np.arange(90., self.fs / 2, self.powerline)

        # append all the bands
        neurobands = np.append(physiological_bands, pathological_bands)

        # index related to each band
        idxbands = map(lambda x: np.where(freqs <= x), neurobands)

        # feature matrix
        featurematrix = np.array([]).reshape(self.n, 0)

        for i in np.arange(1, len(idxbands)):
            # energy at each band
            # divided by sum over all bands (last element of cumulative sum)
            tmp = (cumulative[:, idxbands[i][0][-1]] -
                   cumulative[:, idxbands[i-1][0][-1]]) / cumulative[:, -1]
            featurematrix = np.hstack((featurematrix, tmp.reshape(self.n, 1)))

        return featurematrix


    def computedwtfeat(self):  # dwt_entropy
        """
        It computes for each time series measure from DWT.
        We use orthogonal discrete mother wavelet, Daubechies order two.
        -------------
        Parameters
            self
        -------------
        Returns
            dwtfeat, a matrix which contains the results obtained through
            wavelet transform
        """
        motherwavelet = pywt.Wavelet('db2')  # Daubechies 2nd order
        max_ = pywt.dwt_max_level(self.p, motherwavelet.dec_len) + 1
        # total wavelet entropy
        dwtfeat = np.array([]).reshape(0, max_)

        for i in range(self.n):
            scaledecomp = pywt.wavedec(self.X[i], 'db2')  # coefficients from wd
            energyperscale = np.array(map(lambda x: np.sum(x)**2, scaledecomp[1:]))
            totalenergy = np.sum(energyperscale)   # total energy
            waveletEntropy = - np.sum(energyperscale/totalenergy *
                                np.log(energyperscale/totalenergy))
            tmp = flatten([energyperscale/totalenergy, waveletEntropy])
            dwtfeat = np.vstack((dwtfeat,
                                np.array(tmp).reshape(1, max_)))

        return dwtfeat


    def computetimefeat(self, thresh):
        """
        It computes for each the time spent over a fixed threshold.
        For each band in the vector neurobands we filter the signal, we remove
        edge effects and then we compute how much time each signal spends over
        a fixed threshold. The threshold values are passed as parameters to the
        function. For further uses, of other epileptic stages, it is recommended
        to change the value of the threshold.
        -------------
        Parameters
            self
            thresh, the value of the thresholds related to the ones in neurobands
        -------------
        Returns
            valueoverthresh, a matrix which contains the results obtained through
            wavelet transform
        """
        # physiological bands
        physiological_bands = np.array([0.5, 1., 4., 8., 13., 30., 70.])
        # higher bands - from high gamma to Nyquist
        pathological_bands = np.arange(90., self.fs / 2, self.powerline)
        neurobands = np.append(physiological_bands, pathological_bands)

        # constant factors
        constants = np.arange(2, 8)

        valueoverthresh = np.array([]).reshape(self.n, 0)

        edge = 500  # we remove 500 points at the beginning and at the end
        # of the signal to reduce edge effects

        for j in range(len(neurobands)-1):

            band = neurobands[j+1] - neurobands[j]
            centralf = 0.5 * (neurobands[j+1] + neurobands[j])
            tmpfilter = bandfilter(self.X, self.fs, centralf, band)  # default options
            meanrec = tmpfilter.mean(axis=1)  # mean for each recording
            stdrec = tmpfilter.std(axis=1)    # std for each recording

            valueoverthresh = np.hstack((valueoverthresh, meanrec.reshape(self.n, 1)))
            valueoverthresh = np.hstack((valueoverthresh, stdrec.reshape(self.n, 1)))

            for c in constants:

                tmptime = ((np.abs(tmpfilter[:, edge:-edge]) > c * thresh[j]).sum(axis=1)
                                    / (self.p -2. * edge))
                valueoverthresh = np.hstack((valueoverthresh, tmptime.reshape(self.n, 1)))

        return valueoverthresh



def split_folderstring(folderpath, x):
    """
    This function is specific for the format of our data, as the spatial features
    are placed in folders
    """
    folder = re.match(folderpath, x, flags=re.IGNORECASE).group(0)
    range_ = re.split(folder, x, flags=re.IGNORECASE)
    start, _ = range_[1].split('.pkl')

    return start


def merge_temporal_features(X, fs, powerline, thresholds):
    """
    Here we call the class which performs the preprocessing step + feature
    extraction step from the time series. In particular we compute the
    fft features, dwt, moments and thresholds over the time series
    -------------
    Parameters
        X, data matrix of dimensions #recordings x length time series
        fs, sampling frequency
        powerline, powerline value
        thresholds, the value of the thresholds related to the ones in neurobands
    -------------
    Returns
        features, the set of features used in classification
    """
    FE = featureExtractionTimeSeries(X, fs, powerline)

    fft = FE.computefftfeat()  # fft features
    dwt = FE.computedwtfeat()  # dwt - shannon measure
    mom = FE.computemoments()  # moments of the distribution
    tem = FE.computetimefeat(thresholds)  # time spent over adaptive thresholds

    return np.hstack((fft, dwt, mom, tem))
