import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


def main():

    """
    We pass as argument the path to the pandas.DataFrame use for learning.
    Each column of this dataframe contains the features of interest. The last
    four columns are related to the (x,y,z)-coordinates, Y label
    We use the StratifiedShuffleSplit in order to get a fair split of learning
    and testing set. We perform the split 50 times. After this, we consider the
    the prediction for the test points. In particular, we assess if the
    misclassified channels are far from the epileptic areas. For doing this we
    resort to the label - (x,y,z) coordinates information. From these values,
    we  compute the mean and std for the (x,y,z) pf the epileptic focus. We then
    assess if the misclassified channels are far from these areas.
    """

    path = "/home/vanessa/DATA_SEEG/PKL_FILE/"
    filepath = "/home/vanessa/DATA_SEEG/PKL_FILE/classificationset.pkl"

    df = pd.read_pickle(filepath)

    # first we consider for each patient the position of the epileptic channels
    position_focus = np.zeros((len(df.index.levels[0]), 6))  # mean and std

    for nn, idx in enumerate(df.index.levels[0]):  # loop over the ID for the patient

        bm = df.loc[idx]["Y"]       # we consider the labels
        bm = bm[:len(bm)/2]         # we have the split (2 times # channels)
        bm = np.where(bm == 1)[0]   # boolean mask for epileptic channels

        xf = df.loc[idx]['xcoor'][bm]
        yf = df.loc[idx]['ycoor'][bm]
        zf = df.loc[idx]['zcoor'][bm]

        focus_position[nn] = np.array([np.mean(xf), np.std(xf), np.mean(yf),
                                       np.std(yf), np.mean(zf), np.std(zf)])

    print(focus_position)

    dffocus = pd.DataFrame(data=focus_position, index=df.index.levels[0], columns=[
                        "meanx", "stdx", "meany", "stdy", "meanz", "stdz"])

    dffocus.to_pickle(path + "focus_position.pkl")


    ############################# CLASSIFICATION ###############################

    # random forest estimator
    RF = RandomForestClassifier(n_estimators=1000)
    param_grid = {'max_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}

    # 50 splits
    splits = 50
    confusion_m_results = []
    fn_distance_results = []
    fp_distance_results = []
    fn_idx_results = []
    fp_idx_results = []
    estimator_list = []

    # to preserve the unbalancedness of the class
    sss_ln_ts = StratifiedShuffleSplit(n_splits=splits, test_size=0.15)

    count = 1

    # total number of channels
    n, _  = df.shape

    # we split in training and learning sets
    for idx_ln, idx_ts in sss_ln_ts.split(df.iloc[:, :-4], df["Y"]):

        # we must consider the indeces. This is done in such a way that we can
        # recover the center - focus for each patient
        print("# split: " + str(count))
        X_ln = X[idx_ln]
        X_ts = X[idx_ts]
        y_ln = y[idx_ln]
        y_ts = y[idx_ts]

        estimator = GridSearchCV(RF, param_grid=param_grid,
                                 cv=3, scoring='f1', n_jobs=-1)
        estimator.fit(X_ln[idx_ln], y_ln[idx_ln])

        y_pred = estimator.predict(X_ts)

        ####################### assess position ###############################

        pred_error = idx_ts[y_pred != y_ts]  # test indeces of wrong prediction

        fp_distance = []  # distance from focus
        fn_distance = []
        fp_idx = []       # index for the false positive
        fn_idx = []       # index for the false negative

        for idx_error in pred_error:

            xx = df.loc[idx_error]["xcoor"]    # x coordinate for the channel
            yy = df.loc[idx_error]["ycoor"]    # y coordinate for the channel
            zz = df.loc[idx_error]["zcoor"]    # z coordinate for the channel
            id_p = df.index[idx_error][0]      # identifier for patient

            # standardized distance from the focus
            stardard_distx = (xx - dffocus.loc[id_p]["xmean"]) /
                                    dffocus.loc[id_p]["xstd"]
            standard_disty = (yy - dffocus.loc[id_p]["ymean"]) /
                                    dffocus.loc[id_p]["ystd"]
            standard_distz = (zz - dffocus.loc[id_p]["zmean"]) /
                                    dffocus.loc[id_p]["zstd"]

            if(y_ts[idx_error] == 1):   # false negative
                fn_distance.append([stardard_distx,
                                    standard_disty, standard_distz])
                fn_idx.append(idx_error)

            else:                      # false positive
                fp_distance.append([stardard_distx,
                                    standard_disty, standard_distz])
                fp_idx.append(idx_error)

        fn_distance = np.array(fn_distance)
        fp_distance = np.array(fp_distance)

        fn_idx = np.array(fn_idx)
        fp_idx = np.array(fp_idx)

        cm = confusion_matrix(y[idx_ts], y_pred)

        confusion_m_results.append(cm)
        fn_distance_results.append(fn_distance)
        fp_distance_results.append(fp_distance)
        fn_idx_results.append(fn_idx)
        fp_idx_results.append(fp_idx)

        count = count + 1

    confusion_m_results = np.array(confusion_m_results)
    fn_distance_results = np.array(fn_distance_results)
    fp_distance_results = np.array(fp_distance_results)

    np.save("confusion_matrix.npy", confusion_m_results)
    np.save("fn_distance.npy", fn_distance_results)
    np.save("fp_distance.npy", fp_distance_results)

    # pkl.dump(estimator_list, open("estimators.pkl", 'wb'))


if __name__ == '__main__':
    main()
