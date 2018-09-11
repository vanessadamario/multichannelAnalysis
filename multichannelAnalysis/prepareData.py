import os
import pickle
import numpy as np
import pandas as pd
from sys import argv
from scipy.io import loadmat


"""
This script is used to save the data in a proper format. We consider the mat
files, the labels and the positions dataset. We then save them in a pickle file.
Which has a number of rows equivalent to the number of channels and a number of
columns equivalent to the length of the time series + Y + ptdindex and the
(x,y,z)-coordinates
"""

# flatten a list
def flatten(x):
    return [y for l in x for y in flatten(l)] \
        if type(x) in (list, np.ndarray) else [x]


def main(argv):
    # path relative to the folder of spatial information
    path_pos = argv[1]  # "./channelPositions/"

    # path relative to the folder of labels
    path_tag = argv[2]  #  "./labels/"

    # path relative to time series and channels name - SEEG data
    path_tmp = argv[3]  #  "./MAT_FILE/"

    # path where to save dataframe
    path_pkl = argv[4]  #  "./PKL_FILE/"

    # id patients for patients in the folder
    listid = os.listdir(path_pos)

    for count, id in enumerate(listid):

        newfolder = path_pkl + id + "/"

        data = id + "_channels.mat"
        chan = id + "_channels_name.mat"

        try:  # if the position file exists, we store the data
            pos = pd.read_csv(path_pos + id + "/spatialInfoPreProc.csv",
                                        index_col=0)
            os.mkdir(newfolder)
        except:
            continue

        timeseries = loadmat(path_tmp + data)["data"]
        print("time series shape", timeseries.shape)
        label = np.array(flatten(loadmat(path_tmp + chan)["alpha_ordered"]))
        print(id)

        # this file does not contain channels with artifacts
        tag = pd.read_csv(path_tag + id + ".csv", index_col=0, header=None)

        loc_data = flatten([np.where(t == label)[0] for t in tag.index])

        loc_pos = flatten([np.where(t.split("-")[0] == np.array(pos.index))[0]
                          for t in tag.index])

        l1, l2, l3 = len(loc_pos), len(loc_data), len(tag.index)

        if l1!=l2 or l1!=l3 or l2!=l3:
            loc_pos = []
            loc_data = []
            index_tag = []

            for idx, t in enumerate(tag.index):
                tmp_data = np.where(t == label)[0]
                tmp_pos = np.where(t.split("-")[0] == np.array(pos.index))[0]

                if len(tmp_data) == 1 and len(tmp_pos) == 1:
                    index_tag.append(idx)
                    loc_pos.append(tmp_pos[0])
                    loc_data.append(tmp_data[0])

        else:
            index_tag = np.arange(l3)

        # the spatial channel has no differential measures
        ptd = (pos.loc[:, "PTD"]).values[loc_pos]  # partial tissue density index
        xcoor = (pos.loc[:, "x"]).values[loc_pos]  # x coordinate
        ycoor = (pos.loc[:, "y"]).values[loc_pos]  # y coordinate
        zcoor = (pos.loc[:, "z"]).values[loc_pos]  # z coordinate

        spacedata = np.vstack((ptd, xcoor, ycoor, zcoor)).T
        spacedf = pd.DataFrame(spacedata, index=tag.index[index_tag],
                                columns=['PTD', 'xcoor', 'ycoor', 'zcoor'])

        tagdf = pd.DataFrame(tag.values[index_tag],
                            index=tag.index[index_tag], columns=["Y"])

        timedf = pd.DataFrame(timeseries[loc_data], index=tag.index[index_tag])

        df = pd.concat([timedf,
                    pd.concat([tagdf, spacedf], axis=1)], axis=1)

        df.to_pickle(newfolder + "data.pkl")
