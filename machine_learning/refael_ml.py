import os
import csv
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from machine_learning.neural_network.nn_activator import FeedForwardNet
from nn_models import NeuralNet3

from bokeh.plotting import figure
from bokeh.io import export_png

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from scipy.stats import zscore

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
        mx = np.vstack([self._time_list[time_idx][key][FTR_IDX] for key in sorted(
            self._time_list[time_idx], key=key_func)])
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

    def get_nan_graphs(self):
        f = open('nan_graphs.csv', 'w')
        w = csv.writer(f)
        w.writerow(['Community', 'Time', 'Color'])
        for t in range(self._len):
            m, lab = self._matrix_for_time(t)
            for community in range(m.shape[0]):
                if np.isnan(m[community, 0]):
                    w.writerow([community, t, lab[community]])
        f.close()

    def get_filtered_mx_and_labels(self):
        fin_matrix, all_labels = self._matrix_for_time(self._len - 1)
        indices = [i for i in range(fin_matrix.shape[0]) if not np.isnan(fin_matrix[i, 0])]
        mx_to_csv = fin_matrix[indices, :]
        labels_to_csv = np.array(all_labels)[indices]
        f = open('feature_matrix.csv', 'w')
        w = csv.writer(f)
        for r in range(mx_to_csv.shape[0]):
            w.writerow([mx_to_csv[r, j] for j in range(mx_to_csv.shape[1])])
        f.close()
        g = open('labels.csv', 'w')
        wr = csv.writer(g)
        for r in range(labels_to_csv.shape[0]):
            wr.writerow([int(labels_to_csv[r])])
        g.close()

    def _learn_xgb_binary(self):
        base_dir = __file__.replace("/", os.sep)
        base_dir = os.path.join(base_dir.rsplit(os.sep, 1)[0], "..")
        if not os.path.exists(os.path.join(base_dir, 'fig', 'machine_learning')):
            os.mkdir(os.path.join(base_dir, 'fig', 'machine_learning'))
        f = open(os.path.join(base_dir, 'fig', 'machine_learning', 'ml_auc.csv'), 'w')
        w = csv.writer(f)
        w.writerow(['time', 'train AUC', 'test AUC'])
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
            w.writerow([str(x) for x in [i, np.mean(auc_train), np.mean(auc_test)]])

    def _learn_xgb_multicolor(self):
        fin_matrix, all_labels = self._matrix_for_time(self._len - 1)
        colors_dict = Counter(all_labels)
        num_classes = len(colors_dict.keys())
        n_colors = [0] * num_classes
        for k, v in colors_dict.items():
            n_colors[int(k)] = v
        # acc_train = []
        # acc_test = []
        # for i in range(self._len):  # i is the time.
        for C in [1e3, 1e4, 1e5]:
            fig = figure(plot_width=1000, plot_height=400, title='AUC by PCA dimension: C = ' + str(int(C)),
                         x_axis_label="Number of components from PCA", y_axis_label='AUC')
            auc_train = []
            auc_test = []
            for dim in range(1, 50, 4):
                for i in [self._len - 1]:
                    # get a matrix of shape (number of graphs, number of features) and the label vector for the graphs:
                    components, labels = self._matrix_for_time(i)
                    if self._params["task"] == "Multiclass1":
                        n_colors = [n_colors[0], sum([n_colors[t] for t in range(1, num_classes)])]
                        num_classes = 2
                    # split the graphs into train, evaluation and test sets.
                    acc_train_temp = []
                    acc_test_temp = []
                    for run in range(20):
                        x_train, x_test, y_train, y_test = train_test_split(components, labels, test_size=0.3)
                        x_train, x_test, y_train, y_test = self._treat_data(x_train, x_test, y_train, y_test, dim)
                        x_train, x_eval, y_train, y_eval = train_test_split(x_train, y_train, test_size=0.1)
                        # dtrain = xgb.DMatrix(x_train, y_train, silent=True)
                        # # weight=[float(sum(n_colors)) / n_colors[int(p)] for p in y_train])
                        # dtest = xgb.DMatrix(x_test, y_test, silent=True)
                        # # weight=[float(sum(n_colors)) / n_colors[int(p)] for p in y_test])
                        # deval = xgb.DMatrix(x_eval, y_eval, silent=True)
                        # # weight=[float(sum(n_colors)) / n_colors[int(p)] for p in y_eval])
                        # # Use the default parameters, except linear booster (tree based boosters are overfitting),
                        # # different L2 regularization term lambda and learning rate eta.
                        # params = {'silent': True, 'booster': 'gblinear', 'lambda': 0.07, 'eta': 0.17,
                        #           'objective': 'binary:logistic'}
                        # # params = {'silent': True, 'booster': 'dart', 'lambda': 0.005, 'eta': 0.4, 'max_depth': 20,
                        # #           'num_round': 10, 'ntree_limit': 1000, 'rate_drop': 0.15, 'sample_type': 'weighted',
                        # #           'objective': 'binary:logistic'}
                        #
                        # clf_xgb = xgb.train(params, dtrain=dtrain, evals=[(dtrain, 'train'), (deval, 'eval')],
                        #                     early_stopping_rounds=10, verbose_eval=False)
                        # y_score_test = clf_xgb.predict(dtest)
                        # y_score_train = clf_xgb.predict(dtrain)
                        # train_predict_truth = []
                        # test_predict_truth = []
                        # for name in range(x_train.shape[0]):
                        #     train_predict_truth.append((y_train[name], y_score_train[name]))
                        # for name in range(x_test.shape[0]):
                        #     test_predict_truth.append((y_test[name], y_score_test[name]))
                        # colors_dict_test = Counter(y_test)
                        # colors_dict_train = Counter(y_train)
                        # n_colors_test = [0] * num_classes
                        # n_colors_train = [0] * num_classes
                        # for k, v in colors_dict_train.items():
                        #     n_colors_train[int(k)] = v
                        # for k, v in colors_dict_test.items():
                        #     n_colors_test[int(k)] = v
                        # acc_train_temp.append(
                        #     sum([(int(t[0]) == int(t[1])) / n_colors_train[int(t[0])] for t in train_predict_truth])
                        #     / num_classes)
                        # acc_test_temp.append(
                        #     sum([(int(t[0]) == int(t[1])) / n_colors_test[int(t[0])] for t in test_predict_truth])
                        #     / num_classes)
                        clf = SVC(C=C, gamma='auto', kernel='linear', coef0=5, degree=2)
                        clf.fit(x_train, y_train)
                        y_score_train = clf.predict(x_train)
                        y_score_test = clf.predict(x_test)
                        try:
                            acc_train_temp.append(roc_auc_score(y_train, y_score_train))
                            acc_test_temp.append(roc_auc_score(y_test, y_score_test))
                        except ValueError:
                            continue
                    auc_train.append(np.mean(acc_train_temp))
                    auc_test.append(np.mean(acc_test_temp))
                    # print('time: ' + str(i) + ', train accuracy: ' + str(acc_train[i]) + ', test accuracy: '
                    # + str(acc_test[i]))
                    print('final time, train AUC: ' + str(auc_train[int((dim - 1)/4)]) + ', test AUC: ' +
                          str(auc_test[int((dim - 1)/4)]))
            fig.circle(list(range(1, 50, 4)), auc_train, size=5, color='green', alpha=0.8, legend='Train')
            fig.circle(list(range(1, 50, 4)), auc_test, size=5, color='red', alpha=0.8, legend='Test')
            fig.legend.location = "top_left"
            export_png(fig, os.path.join("fig", "machine_learning", "AUC_vs_PCA_C_" + str(int(C)) + ".png"))

    def _treat_data(self, x_train, x_test, y_train, y_test, dim):
        new_x_train = x_train
        new_x_test = x_test
        new_y_train = y_train
        new_y_test = y_test

        if self._params["task"] != "Binary":
            small_value = 10 ** -6
            x_train_before_dim_lowering = -np.log10(x_train[:, 1:] + small_value)
            x_test_before_dim_lowering = -np.log10(x_test[:, 1:] + small_value)
            # x_train_before_dim_lowering = x_train
            # x_test_before_dim_lowering = x_test

            # Z scoring
            for i in range(x_train_before_dim_lowering.shape[1]):
                if x_train_before_dim_lowering[:, i].std() > 0:
                    x_train_before_dim_lowering[:, i] = \
                        (x_train_before_dim_lowering[:, i] - x_train_before_dim_lowering[:, i].mean()) / \
                        x_train_before_dim_lowering[:, i].std()
            for i in range(x_test_before_dim_lowering.shape[1]):
                if x_test_before_dim_lowering[:, i].std() > 0:
                    x_test_before_dim_lowering[:, i] = \
                        (x_test_before_dim_lowering[:, i] - x_test_before_dim_lowering[:, i].mean()) / \
                        x_test_before_dim_lowering[:, i].std()

            # PCA:
            pca_train = PCA(n_components=dim)  # OR FEATURE SELECTION USING TRAIN LABELS SPEARMAN CORRELATION.
            pca_test = PCA(n_components=dim)
            fit_pca_train = pca_train.fit(np.transpose(x_train_before_dim_lowering))
            fit_pca_test = pca_test.fit(np.transpose(x_test_before_dim_lowering))
            x_train_after_dim_lowering = np.transpose(fit_pca_train.components_)
            x_test_after_dim_lowering = np.transpose(fit_pca_test.components_)

            # Feature Selection
            # top_k = 150
            # corrs = []
            # for col in range(x_train_before_dim_lowering.shape[1]):
            #     if np.std(x_train_before_dim_lowering[:, col]) == 0:
            #         corrs.append(0)
            #         continue
            #     corr, _ = spearmanr(x_train_before_dim_lowering[:, col], new_y_train)
            #     corrs.append(corr)
            # features_sorted = np.argsort(- np.abs(corrs))
            # features_selected = sorted([features_sorted[i] for i in range(min(len(features_sorted), top_k))])  # Doesn't have to be sorted.
            # x_train_after_dim_lowering = x_train_before_dim_lowering[:, features_selected]
            # x_test_after_dim_lowering = x_test_before_dim_lowering[:, features_selected]

            new_x_train = x_train_after_dim_lowering
            new_x_test = x_test_after_dim_lowering

        return new_x_train, new_x_test, new_y_train, new_y_test

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
