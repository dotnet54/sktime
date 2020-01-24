from sklearn import datasets

from sktime.contrib.ts_chief.tschief import TSChiefForest
from sktime.contrib.ts_chief.shifaz_dev.test import *
from sklearn.metrics import accuracy_score


def test_chief():
    x_train, y_train, x_test, y_test = load_dataset('Coffee')
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    print(f'Train: {x_train.shape}, Test: {x_test.shape}')

    model = TSChiefForest(num_trees=2,
                          num_similarity_candidate_splits=3,
                          num_dictionary_candidate_splits=0,
                          num_interval_candidate_splits=0)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test, y=y_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {score}')



test_chief()

def test_RF():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier

    np.random.seed(0)
    iris_X, iris_y = datasets.load_iris(return_X_y=True)
    indices = np.random.permutation(len(iris_X))
    iris_X_train = iris_X[indices[:-10]]
    iris_y_train = iris_y[indices[:-10]]
    iris_X_test = iris_X[indices[-10:]]
    iris_y_test = iris_y[indices[-10:]]
    # Create and fit a nearest-neighbor classifier

    knn = KNeighborsClassifier()
    knn.fit(iris_X_train, iris_y_train)
    knn_pred = knn.predict(iris_X_test)


    dt = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
    dt.fit(iris_X_train, iris_y_train)
    dt_pred = dt.predict(iris_X_test)

    rf = RandomForestClassifier(n_estimators=5, max_depth=None,min_samples_split=2, random_state=0)
    rf.fit(iris_X_train, iris_y_train)
    rf_pred = rf.predict(iris_X_test)

    print('done')


# test_RF()