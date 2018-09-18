import pickle
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
    focus_pos_path = "/home/vanessa/DATA_SEEG/PKL_FILE/focus_position.pkl"
    class_set_path = "/home/vanessa/DATA_SEEG/PKL_FILE/finalclassificationset.pkl"

    dffocus = pd.read_pickle(focus_pos_path)
    df = pd.read_pickle(class_set_path)  # dataframe which contains the features

    # random forest estimator
    RF = RandomForestClassifier(n_estimators=1000)
    param_grid = {'max_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}

    # 50 splits
    splits = 1
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

    learning_indexes = []
    test_indexes = []
    predictions = []

    # we split in training and learning sets
    X  = df.iloc[:, :-4].values
    y = df["Y"].values

    print(X.shape, y.shape)

    for idx_ln, idx_ts in sss_ln_ts.split(X, y):

        learning_indexes.append(idx_ln)
        test_indexes.append(idx_ts)

        # we must consider the indeces. This is done in such a way that we can
        # recover the center - focus for each patient
        print("# split: " + str(count))
        X_ln = X[idx_ln]
        X_ts = X[idx_ts]
        y_ln = y[idx_ln]
        y_ts = y[idx_ts]

        estimator = GridSearchCV(RF, param_grid=param_grid,
                                 cv=3, scoring='f1', n_jobs=-1)
        estimator.fit(X_ln, y_ln)

        y_pred = estimator.predict(X_ts)
        predictions.append(y_pred)

        ####################### assess position ###############################

        pred_error = idx_ts[y_pred != y_ts]  # test indexes of wrong prediction

        fp_distance = []  # distance from focus
        fn_distance = []
        fp_idx = []       # index for the false positive
        fn_idx = []       # index for the false negative

        for idx_error in pred_error:

            xx = df.iloc[idx_error]["xcoor"]    # x coordinate for the channel
            yy = df.iloc[idx_error]["ycoor"]    # y coordinate for the channel
            zz = df.iloc[idx_error]["zcoor"]    # z coordinate for the channel
            id_p = df.index[idx_error][0]      # identifier for patient

            # standardized distance from the focus
            stardard_distx = np.abs(xx - dffocus.loc[id_p]["meanx"]) / dffocus.loc[id_p]["stdx"]
            standard_disty = np.abs(yy - dffocus.loc[id_p]["meany"]) / dffocus.loc[id_p]["stdy"]
            standard_distz = np.abs(zz - dffocus.loc[id_p]["meanz"]) / dffocus.loc[id_p]["stdz"]

            if(y[idx_error] == 1):   # false negative
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

    np.save(path + "_results/confusion_matrix.npy", confusion_m_results)
    pickle.dump(fn_distance_results,
            open(path + "_results/fn_distance.pkl", 'wb'))
    pickle.dump(fp_distance_results,
            open(path + "_results/fp_distance.pkl", 'wb'))



    pickle.dump(learning_indexes,
            open(path + "_results/learning_indexes.pkl", 'wb'))
    pickle.dump(test_indexes,
            open(path + "_results/testing_indexes.pkl", 'wb'))
    pickle.dump(predictions,
            open(path + "_results/predictions.pkl", 'wb'))


if __name__ == '__main__':
    main()
