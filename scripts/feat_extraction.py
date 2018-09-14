import re
import os
import numpy as np
import pandas as pd
from multichannelAnalysis.featureExtraction import remove_powerline
from multichannelAnalysis.featureExtraction import merge_temporal_features


def main():

    """
    argv[1] is the path to the main folder
    It contains subfolders, one for each
    patient. This are such that, in each folder, which has its own identifier,
    there is a pickle file, which contains the data organized in a data frame
    argv[2] is the name of the file contained in the subfolder. The name of
    this file is common across all patients
    example : argv[1] "/home/../"
              argv[2] "/data.pkl"
    """

    pathfolder = "/home/vanessa/DATA_SEEG/PKL_FILE/"
    filename = "/data.pkl"
    # pathfolder = argv[1]
    # filename = argv[2]

    ti = 10.                 # initial time
    tf = 590.                # final time
    t_split = 300.           # split
    fs = 1000.               # sampling frequency
    powerline = 50.

    thresholds = np.load("threshold.npy")    # load the threshold file
    meanthresh = thresholds.mean(axis=0)[1::2]
    stdthresh = thresholds.std(axis=0)[1::2]

    # features = 159  # classification features + (x,y,z)-coordinates

    for ii, id in enumerate(os.listdir(pathfolder)):

        print(id)

        df = pd.read_pickle(pathfolder + id + filename)

        validchannels = np.where(~df.loc[:, "PTD"].isnull())[0]  # remove NaN values

        df = df.iloc[validchannels, :]
        _, p = df.shape

        timeseries = df.values[:, :-5]  # we are not considering Y, ptd, coordinates

        data = remove_powerline(timeseries, fs)     # remove power line effects

        #################### split into 2 fragments ############################

        split1half = data[:, int(fs*ti):int(fs*t_split)]
        split2half = data[:, int(fs*t_split):int(fs*tf)]

        timefeat1half = merge_temporal_features(split1half, fs, powerline,
                                                meanthresh)
        timefeat2half = merge_temporal_features(split2half, fs, powerline,
                                                meanthresh)

        ########################################################################

        cc = [df.index[t] for t in range(len(df.index))]
        arrays = [[id]*(2*len(df.index)), cc + cc]

        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=['patient', 'channel'])

        # temporal features from SEEG
        timefeatdf = pd.DataFrame(data=np.vstack((timefeat1half,
                                        timefeat2half)), index=index)

        # spatial features for MRI
        spacefeat = df.values[:, -4:]
        spacefeatdf = pd.DataFrame(data=np.vstack((spacefeat, spacefeat)),
                                    index=index, columns=
                                    ['PTD', 'xcoor', 'ycoor', 'zcoor'])

        # y labels
        ylab = df.values[:, -5]
        Ylabel = pd.DataFrame(data=np.append(ylab, ylab), index=index,
                            columns=["Y"])

        # pickle file in output
        outputpkl = pd.concat([timefeatdf, spacefeatdf, Ylabel], axis=1)

        outputpkl.to_pickle(pathfolder + id + "/features.pkl")

        if ii == 0:
            ddd = outputpkl
        else:
            ddd = pd.concat([ddd, outputpkl], axis=0)

    ddd.to_pickle(pathfolder + "classificationset.pkl")
    

if __name__ == '__main__':
    main()
