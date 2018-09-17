from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def main():

    # load file
    filepath = "/home/vanessa/DATA_SEEG/PKL_FILE/classificationset.pkl"
    dd = pd.read_pickle(filepath)

    idlist = [f for f in dd.index.levels[0] if len(dd.loc[f].index)>0]

    for id in idlist:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        print(dd.loc[id].shape)

        if len(dd.loc[id].index) > 0: # not empty dataframe

            bm = dd.loc[id]["Y"] == 1

            xp = dd.loc[id]["xcoor"][bm]
            yp = dd.loc[id]["ycoor"][bm]
            zp = dd.loc[id]["zcoor"][bm]

            xn = dd.loc[id]["xcoor"][~bm]
            yn = dd.loc[id]["ycoor"][~bm]
            zn = dd.loc[id]["zcoor"][~bm]

            ax.scatter(xp, yp, zp, c="C0", label="focus")
            ax.scatter(xn, yn, zn, c="C1")
            plt.legend()

            plt.show()
            plt.close()


if __name__ == '__main__':
    main()
