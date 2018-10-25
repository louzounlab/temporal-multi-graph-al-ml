from enum import Enum
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from machine_learning.neural_network.nn_activator import FeedForwardNet
from nn_models import NeuralNet3
FTR_IDX = 0
LABEL_IDX = 1


class RefaelML:
    def __init__(self, params, time_list):
        self._time_list = time_list
        self._len = len(time_list)
        self._method = params['learn_method'] if 'learn_method' in params else "nn"
        self._params = params

    def run(self):
        if self._method.value == "nn":
            self._neural_net()
        if self._method.value == "XG_Boost":
            self._learn_xgb()

    def _matrix_for_time(self, time_idx: int, key_func=None):
        mx = np.vstack([self._time_list[time_idx][key][FTR_IDX] for key in sorted(self._time_list[time_idx], key=key_func)])
        labels = [self._time_list[time_idx][key][LABEL_IDX] for key in sorted(self._time_list[time_idx], key=key_func)]
        return mx, labels

    def _dict_for_time(self, time_idx: int):
        ftr = {}
        labels = {}
        for name, ftr_tupple in self._time_list[time_idx].items():
            ftr[name] = ftr_tupple[FTR_IDX]
            labels[name] = ftr_tupple[LABEL_IDX]
        return ftr, labels

    def _neural_net(self):
        early_stop = self._params['early_stop'] if 'early_stop' in self._params else 200
        epochs = self._params['epochs'] if 'epochs' in self._params else 300

        model = NeuralNet3(layers_size=(225, 150, 75), lr=0.001)
        network = FeedForwardNet(model, train_size=0.7, gpu=False)

        for i in range(self._len):
            print("-----------------------------------    TIME " + str(i) + "    -------------------------------------")
            components, labels = self._dict_for_time(i)
            network.update_data(components, labels)
            network.train(epochs, early_stop=early_stop, validation_rate=200)
            network.test()

    def _learn_xgb(self):
        for i in range(self._len):  # i is the time.
            # get a matrix of shape (number of graphs, number of features) and the label vector for the graphs:
            components, labels = self._matrix_for_time(i)
            auc_train = []
            auc_test = []
            for num_splits in range(500):
                # split the graphs into train, evaluation and test sets.
                x_train, x_test, y_train, y_test = train_test_split(components, labels, train_size=0.7)
                x_train, x_eval, y_train, y_eval = train_test_split(x_train, y_train, test_size=0.1)
                dtrain = xgb.DMatrix(x_train, y_train, silent=True)
                dtest = xgb.DMatrix(x_test, y_test, silent=True)
                deval = xgb.DMatrix(x_eval, y_eval, silent=True)
                # Use the default parameters, except linear booster (tree based boosters are overfitting),
                # different L2 regularization term lambda and learning rate eta.
                params = {'silent': True, 'booster': 'gblinear', 'lambda': 0.07, 'eta': 0.17,
                          'objective': 'binary:logistic'}
                clf_xgb = xgb.train(params, dtrain=dtrain, evals=[(dtrain, 'train'), (deval, 'eval')],
                                    early_stopping_rounds=10, verbose_eval=False)
                y_score_test = clf_xgb.predict(dtest)
                y_score_train = clf_xgb.predict(dtrain)
                # roc_auc_score can't handle with one class only:
                try:
                    r1 = roc_auc_score(y_test, y_score_test)
                except ValueError:
                    continue
                try:
                    r2 = roc_auc_score(y_train, y_score_train)
                except ValueError:
                    continue
                auc_test.append(r1)
                auc_train.append(r2)
            print('time: ' + str(i) + ', train_AUC: ' + str(np.mean(auc_train)) + ', test_AUC: ' +
                  str(np.mean(auc_test)))


