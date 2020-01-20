
from sktime.classifiers.shifaz.tschief import ChiefTree
from sktime.classifiers.shifaz.test import *
from sklearn.metrics import accuracy_score

def test_chief():
    X_train, y_train, X_test, y_test = load_dataset('Coffee')
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    print(f'Train: {X_train.shape}, Test: {X_test.shape}')

    params = {
        'k' : 1,
        'Ce' : 5,
        'Cb' : 0
    }
    model = ChiefTree(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test, y_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {score}')

test_chief()

