
from sktime.classifiers.shifaz.tschief import ChiefTree
from sktime.classifiers.shifaz.test import *
from sklearn.metrics import accuracy_score

def test_chief():
    print('loading dataset')

    X_train, y_train, X_test, y_test = load_dataset('ItalyPowerDemand')
    print(f'Train: {X_train.shape}, Test: {y_train.shape}')

    params = {
        'C_e' : 5,
        'C_b' : 5
    }
    model = ChiefTree(params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test, y_test)
    score = accuracy_score(y_test, y_pred)

    print(f'Accuracy: {score}')

test_chief()

