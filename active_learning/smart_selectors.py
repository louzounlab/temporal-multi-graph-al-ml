from operator import itemgetter
import numpy as np
from scipy.spatial.distance import cdist
import xgboost as xgb
from sklearn.model_selection import train_test_split
from nn_activator import FeedForwardNet
from nn_models import NeuralNet3, ActiveLearningModel
import random


class SmartSelector:
    def __init__(self, params, batch_size=1, ml_method="nn", distance_method="euclidean", num_classes=None):
        self._params = params
        self._ml_type = ml_method
        self._dist_type = distance_method
        self._batch_size = batch_size
        self._num_classes = num_classes if num_classes is not None else None

    def distance_select(self, data1, data2):
        if self._dist_type == "euclidean":
            return self._euclidean(data1, data2, typ="euclidean")

    def ml_select(self, train, test):
        if self._ml_type == "nn":
            return self._neural_net(train, test)
        elif self._ml_type == "XG_Boost":
            return self._xg_boost(train, test)
        elif self._ml_type == "rand":
            return self._rand(train, test)

    @staticmethod
    def _to_matrix(data):
        mx = np.vstack([data[name][0] for name in sorted(data)])
        labels = [data[name][1] for name in sorted(data)]
        idx_to_name = [name for name in sorted(data)]
        return mx, idx_to_name, labels

    @staticmethod
    def _sep_dict(in_data: dict):
        data = {}
        labels = {}
        for name, (vec, label) in in_data.items():
            data[name] = vec
            labels[name] = label
        return data, labels

    def _rand(self, data1, data2):
        mx2, idx_2, l2 = self._to_matrix(data2)
        name_to_idx = {name: i for i, name in enumerate(idx_2)}
        random.shuffle(idx_2)
        guess = [(i, random.randint(0, self._num_classes - 1), l2[name_to_idx[i]]) for i in
                 idx_2] if self._num_classes is not None else [
            (i, 0 if np.random.normal(0.5, 0.1) > 0.5 else 1, l2[name_to_idx[i]]) for i in idx_2]
        guess.sort(key=itemgetter(1), reverse=True)
        stop = min(self._batch_size, len(idx_2))
        return guess[0:stop]

    def _euclidean(self, data1, data2, typ="euclidean"):
        mx1, idx_1, l1 = self._to_matrix(data1)
        mx2, idx_2, l2 = self._to_matrix(data2)
        # Get euclidean distances as 2D array
        dists = cdist(mx1, mx2, typ)
        # return the most distant rows
        top_index = dists.mean(axis=0).argsort(kind='heapsort').tolist()
        stop = min(self._batch_size, len(top_index))
        selected_idx = top_index[0:stop]
        return [idx_2[i] for i in selected_idx]

    def _neural_net(self, train, test):

        early_stop = self._params['early_stop'] if 'early_stop' in self._params else 102
        epochs = self._params['epochs'] if 'epochs' in self._params else 300
        if not hasattr(self, '_network'):
            model = NeuralNet3(layers_size=(225, 150, 75), lr=0.001)
            self._binary_neural_network = FeedForwardNet(model, train_size=0.8, gpu=False)

        train_data, train_label = self._sep_dict(train)
        self._binary_neural_network.update_data(train_data, train_label)
        self._binary_neural_network.train(epochs, early_stop=early_stop, validation_rate=200, stop_auc=5)
        clean_test = {name: tup[0] for name, tup in test.items()}         # remove labels from the test
        results = self._binary_neural_network.predict(clean_test)
        all_data = [(key, val.item(), test[key][1]) for key, val in sorted(results.items(), key=lambda x: x[1],
                                                                           reverse=True if not self._params[
                                                                               'white_label'] else False)]
        stop = min(self._batch_size, len(all_data))
        return all_data[0:stop]

    def _xg_boost(self, train, test):
        if self._params["task"] == "Binary":
            return self._xgb_bi(train, test)
        else:
            return self._xgb_multi(train, test)

    def _xgb_bi(self, train, test):
        train_mx, train_idx, y_train = self._to_matrix(train)
        test_mx, test_idx, y_test = self._to_matrix(test)
        train_mx, eval_mx, y_train, y_eval = train_test_split(train_mx, y_train, test_size=0.1)
        dtrain = xgb.DMatrix(train_mx, y_train, silent=True)
        deval = xgb.DMatrix(eval_mx, y_eval, silent=True)
        dtest = xgb.DMatrix(test_mx, silent=True)
        params = {'silent': True, 'booster': 'gblinear', 'lambda': 0.07, 'eta': 0.17, 'objective': 'binary:logistic'}
        clf_xgb = xgb.train(params, dtrain=dtrain, evals=[(dtrain, 'train'), (deval, 'eval')],
                            early_stopping_rounds=10, verbose_eval=False)
        try:
            y_score_test = clf_xgb.predict(dtest)
        except ValueError:
            y_score_test = 0
        index_predict = [(test_idx[i], y_score_test[i], y_test[i]) for i in range(len(test_idx))]
        index_predict.sort(key=itemgetter(1), reverse=True if not self._params['white_label'] else False)
        stop = min(self._batch_size, len(index_predict))
        return [index_predict[i] for i in range(stop)]

    def _xgb_multi(self, train, test):
        train_mx, train_idx, y_train = self._to_matrix(train)
        test_mx, test_idx, y_test = self._to_matrix(test)
        train_mx, eval_mx, y_train, y_eval = train_test_split(train_mx, y_train, test_size=0.1)
        dtrain = xgb.DMatrix(train_mx, y_train, silent=True)
        deval = xgb.DMatrix(eval_mx, y_eval, silent=True)
        dtest = xgb.DMatrix(test_mx, silent=True)
        params = {'silent': True, 'booster': 'gblinear', 'lambda': 0.07, 'eta': 0.17,
                  'objective': 'multi:softprob', 'num_class': 4 if self._params["task"] == "Multiclass1" else 8}
        clf_xgb = xgb.train(params, dtrain=dtrain, evals=[(dtrain, 'train'), (deval, 'eval')],
                            early_stopping_rounds=10, verbose_eval=False)
        y_score_test = clf_xgb.predict(dtest)
        index_predict_prob = []
        for i in range(len(test_idx)):
            j = np.argmax(y_score_test[i, :])
            index_predict_prob.append((test_idx[i], j, y_test[i], y_score_test[i, j]))
        index_predict_prob.sort(key=itemgetter(3), reverse=True)
        index_predict = [(x, y, z) for x, y, z, w in index_predict_prob]
        stop = min(self._batch_size, len(index_predict))
        to_return = [index_predict[batch] for batch in range(stop)]
        return to_return
