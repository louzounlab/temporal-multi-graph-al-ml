import os
import csv
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
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
        if self._method == "nn":
            self._neural_net()
        if self._method == "XG_Boost":
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
        if self._params["task"] == "Binary":
            self._learn_xgb_binary()
        else:
            self._learn_xgb_multicolor()

    def _learn_xgb_binary(self):
        for i in range(self._len):  # i is the time.
            # get a matrix of shape (number of graphs, number of features) and the label vector for the graphs:
            components, labels = self._matrix_for_time(i)
            auc_train = []
            auc_test = []
            for num_splits in range(100):
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

    def _learn_xgb_multicolor(self):
        fin_matrix, all_labels = self._matrix_for_time(self._len - 1)
        colors_dict = Counter(all_labels)
        num_classes = len(colors_dict.keys())
        n_colors = [0] * num_classes
        for k, v in colors_dict.items():
            n_colors[int(k)] = v
        acc_train = []
        acc_test = []
        for i in range(self._len):  # i is the time.
            # get a matrix of shape (number of graphs, number of features) and the label vector for the graphs:
            components, labels = self._matrix_for_time(i)
            # split the graphs into train, evaluation and test sets.
            acc_train_temp = []
            acc_test_temp = []
            for run in range(100):
                x_train, x_test, y_train, y_test = train_test_split(components, labels, train_size=0.7)
                x_train, x_eval, y_train, y_eval = train_test_split(x_train, y_train, test_size=0.1)
                dtrain = xgb.DMatrix(x_train, y_train, silent=True,
                                     weight=[float(sum(n_colors)) / n_colors[int(p)] for p in y_train])
                dtest = xgb.DMatrix(x_test, y_test, silent=True,
                                    weight=[float(sum(n_colors)) / n_colors[int(p)] for p in y_test])
                deval = xgb.DMatrix(x_eval, y_eval, silent=True,
                                    weight=[float(sum(n_colors)) / n_colors[int(p)] for p in y_eval])
                # Use the default parameters, except linear booster (tree based boosters are overfitting),
                # different L2 regularization term lambda and learning rate eta.
                params = {'silent': True, 'booster': 'gblinear', 'lambda': 0.07, 'eta': 0.17,
                          'objective': 'multi:softmax', 'num_class': num_classes}
                clf_xgb = xgb.train(params, dtrain=dtrain, evals=[(dtrain, 'train'), (deval, 'eval')],
                                    early_stopping_rounds=10, verbose_eval=False)
                y_score_test = clf_xgb.predict(dtest)
                y_score_train = clf_xgb.predict(dtrain)
                train_predict_truth = []
                test_predict_truth = []
                for name in range(x_train.shape[0]):
                    train_predict_truth.append((y_train[name], y_score_train[name]))
                for name in range(x_test.shape[0]):
                    test_predict_truth.append((y_test[name], y_score_test[name]))
                colors_dict_test = Counter(y_test)
                colors_dict_train = Counter(y_train)
                n_colors_test = [0] * num_classes
                n_colors_train = [0] * num_classes
                for k, v in colors_dict_train.items():
                    n_colors_train[int(k)] = v
                for k, v in colors_dict_test.items():
                    n_colors_test[int(k)] = v
                acc_train_temp.append(
                    sum([(int(t[0]) == int(t[1])) / n_colors_train[int(t[0])] for t in train_predict_truth])
                    / num_classes)
                acc_test_temp.append(
                    sum([(int(t[0]) == int(t[1])) / n_colors_test[int(t[0])] for t in test_predict_truth])
                    / num_classes)
            acc_train.append(np.mean(acc_train_temp))
            acc_test.append(np.mean(acc_test_temp))
            print('time: ' + str(i) + ', train accuracy: ' + str(acc_train[i]) + ', test accuracy: ' + str(acc_test[i]))
        if not os.path.exists(os.path.join(os.getcwd(), 'log_no_log')):
            os.mkdir('log_no_log')
        f = open(os.path.join(os.getcwd(), 'log_no_log', 'all_log.csv'), 'w')
        w = csv.writer(f)
        w.writerow(['time', 'train accuracy', 'test accuracy'])
        for t, acc_trn, acc_tst in zip(range(self._len), acc_train, acc_test):
            w.writerow([t, acc_trn, acc_tst])
        f.close()

    def grid_xgb_binary(self, param_combinations):  # param_combs: a list of dictionaries {param_str: param_value}.
        if not os.path.exists(os.path.join(os.getcwd(), 'parameter_check')):
            os.mkdir('parameter_check')
        # train percentage
        components_labels = [self._matrix_for_time(i) for i in [int(t * (self._len - 1) / 4) for t in range(5)]]
        for train_p in [30, 50, 70]:
            f = open(os.path.join(os.getcwd(), 'parameter_check', "results_train_p" + str(train_p) + ".csv"), 'w')
            w = csv.writer(f)
            w.writerow([str(p) for p in param_combinations[0].keys()] +
                       ['train_AUC_time_quarter' + str(q) for q in range(5)]
                       + ['test_AUC_time_quarter' + str(q) for q in range(5)])
            for param_comb in param_combinations:
                train_auc_q = []
                test_auc_q = []
                for time in range(5):
                    auc_train = []
                    auc_test = []
                    for num_splits in range(100):
                        X_train, X_test, y_train, y_test = train_test_split(
                            components_labels[time][0], components_labels[time][1], test_size=1 - float(train_p) / 100)
                        X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.1)
                        dtrain = xgb.DMatrix(X_train, y_train, silent=True)
                        dtest = xgb.DMatrix(X_test, y_test, silent=True)
                        deval = xgb.DMatrix(X_eval, y_eval, silent=True)
                        params = param_comb.copy()
                        ntree_limit = param_comb['ntree_limit'] if 'ntree_limit' in param_comb.keys() else None
                        if 'ntree_limit' in param_comb.keys():
                            del params['ntree_limit']
                        early_stopping_rounds = param_comb[
                            'early_stopping_rounds'] if 'early_stopping_rounds' in param_comb.keys() else None
                        if 'early_stopping_rounds' in param_comb.keys():
                            del params['early_stopping_rounds']
                        if early_stopping_rounds is not None:
                            clf_xgb = xgb.train(params, dtrain=dtrain, evals=[(dtrain, 'train'), (deval, 'eval')],
                                                early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
                        else:
                            clf_xgb = xgb.train(params, dtrain=dtrain, evals=[(dtrain, 'train'), (deval, 'eval')],
                                                early_stopping_rounds=10, verbose_eval=False)
                        if ntree_limit is not None:
                            y_score_test = clf_xgb.predict(dtest, ntree_limit=ntree_limit)
                            y_score_train = clf_xgb.predict(dtrain, ntree_limit=ntree_limit)
                        else:
                            y_score_test = clf_xgb.predict(dtest, ntree_limit)
                            y_score_train = clf_xgb.predict(dtrain, ntree_limit)
                        # ROC AUC has a problem with only one class
                        try:
                            r1 = roc_auc_score(y_test, y_score_test)
                        except ValueError:
                            continue
                        auc_test.append(r1)

                        try:
                            r2 = roc_auc_score(y_train, y_score_train)
                        except ValueError:
                            continue
                        auc_train.append(r2)
                    train_auc_q.append(np.mean(auc_train))
                    test_auc_q.append(np.mean(auc_test))
                w.writerow([str(param_comb[p]) for p in param_comb.keys()] +
                           [str(q) for q in train_auc_q] + [str(q) for q in test_auc_q])
        return None

    def grid_xgb_multicolor(self, param_combinations):  # param_combinations: a list of dictionaries {param_str: param_value}.
        if not os.path.exists(os.path.join(os.getcwd(), 'parameter_check')):
            os.mkdir('parameter_check')
        _, all_labels = self._matrix_for_time(self._len - 1)
        colors_dict = Counter(all_labels)
        num_classes = len(colors_dict.keys())
        n_colors = [0] * num_classes
        for k, v in colors_dict.items():
            n_colors[int(k)] = v
        components_labels = [self._matrix_for_time(i) for i in [int(t * (self._len - 1) / 4) for t in range(5)]]
        for train_p in [30, 50, 70]:
            f = open(
                os.path.join(os.getcwd(), 'parameter_check', "results_train_p" + str(train_p) + ".csv"), 'w')
            w = csv.writer(f)
            w.writerow([str(p) for p in param_combinations[0].keys()] +
                       ['train_AUC_time_quarter' + str(q) for q in range(5)]
                       + ['test_AUC_time_quarter' + str(q) for q in range(5)])
            for param_comb in param_combinations:
                train_acc_q = []
                test_acc_q = []
                for time in range(5):
                    acc_train = []
                    acc_test = []
                    for num_splits in range(100):
                        X_train, X_test, y_train, y_test = train_test_split(
                            components_labels[time][0], components_labels[time][1], train_size=float(train_p) / 100)
                        X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.1)
                        dtrain = xgb.DMatrix(X_train, y_train, silent=True,
                                             weight=[1 / n_colors[int(p)] for p in y_train])
                        dtest = xgb.DMatrix(X_test, y_test, silent=True, weight=[1 / n_colors[int(p)] for p in y_test])
                        deval = xgb.DMatrix(X_eval, y_eval, silent=True, weight=[1 / n_colors[int(p)] for p in y_eval])
                        params = param_comb.copy()
                        ntree_limit = param_comb['ntree_limit'] if 'ntree_limit' in param_comb.keys() else None
                        if 'ntree_limit' in param_comb.keys():
                            del params['ntree_limit']
                        early_stopping_rounds = param_comb['early_stopping_rounds'] if 'early_stopping_rounds' in \
                                                                                       param_comb.keys() else None
                        if 'early_stopping_rounds' in param_comb.keys():
                            del params['early_stopping_rounds']
                        if early_stopping_rounds is not None:
                            clf_xgb = xgb.train(params, dtrain=dtrain, evals=[(dtrain, 'train'), (deval, 'eval')],
                                                early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
                        else:
                            clf_xgb = xgb.train(params, dtrain=dtrain, evals=[(dtrain, 'train'), (deval, 'eval')],
                                                early_stopping_rounds=10, verbose_eval=False)
                        if 'ntree_limit' in param_comb.keys():
                            y_score_test = clf_xgb.predict(dtest, ntree_limit=ntree_limit)
                            y_score_train = clf_xgb.predict(dtrain, ntree_limit=ntree_limit)
                        else:
                            y_score_test = clf_xgb.predict(dtest)
                            y_score_train = clf_xgb.predict(dtrain)
                        train_predict_truth = []
                        test_predict_truth = []
                        for name in range(X_train.shape[0]):
                            color = np.argmax(y_score_train[name, :])
                            train_predict_truth.append((y_train[name], color))
                        for name in range(X_test.shape[0]):
                            color = np.argmax(y_score_test[name, :])
                            test_predict_truth.append((y_test[name], color))
                        acc_train.append(accuracy_score([x for x, y in train_predict_truth],
                                                        [y for x, y in train_predict_truth],
                                                        [1/sum([t == i for t in train_predict_truth])
                                                         for i in train_predict_truth]))
                        acc_test.append(accuracy_score([x for x, y in test_predict_truth],
                                                       [y for x, y in test_predict_truth],
                                                       [1/sum([t == i for t in test_predict_truth])
                                                        for i in test_predict_truth]))
                        # acc_train.append(accuracy_score([x for x, y in train_predict_truth],
                        #                                 [y for x, y in train_predict_truth]))
                        # acc_test.append(accuracy_score([x for x, y in test_predict_truth],
                        #                                [y for x, y in test_predict_truth]))
                    train_acc_q.append(np.mean(acc_train))
                    test_acc_q.append(np.mean(acc_test))
                w.writerow([str(param_comb[p]) for p in param_comb.keys()] +
                           [str(q) for q in train_acc_q] + [str(q) for q in test_acc_q])
                print(["Quarter: " + str(q) + ", Train: " + str(train_acc_q[q]) + ", Test: " + str(test_acc_q[q]) for q
                       in range(len(train_acc_q))])
        return None
