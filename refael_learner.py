import os
import numpy as np
from active_learning_binary import TimedActiveLearningBi
from active_learning_multiclass import TimedActiveLearningMulti
from data_loader import DataLoader
from parameters import REFAEL_PARAM, ACTIVE_LEARNING, MACHINE_LEARNING
from refael_ml import RefaelML
# from grid_parameters import param_grid


class RefaelLearner:
    def __init__(self):
        self._params = REFAEL_PARAM
        self._params['database_full_name'] = self._full_name()
        self._base_dir = __file__.replace("/", os.sep)
        self._base_dir = os.path.join(self._base_dir.rsplit(os.sep, 1)[0])
        self._data_loader = DataLoader(self._params)

    def _full_name(self):
        if self._params["task"] == 'Binary':
            task = "bi"
        elif self._params["task"] == "Multiclass1":
            task = "c1"
        else:
            task = "c2"
        return self._params['database'] + "__" + task + "_ds_" + str(self._params['days_split']) + "_st_" + \
            str(self._params['start_time']) + "_ws_" + str(self._params['window_size']) + "_d_" + \
            str(self._params['directed']) + "_mc_" + str(self._params['max_connected'])

    def data_loader(self):
        return self._data_loader

    def base_dir(self):
        return self._base_dir

    def run_ml(self):
        # Simple learning over all graphs in a given time. Returning a .csv file of AUC values.
        if self._params['task'] == 'Binary':
            data = self._data_loader.filter_by_nodes(min_nodes=self._params['min_nodes']) if self._params['min_nodes'] \
                else self._data_loader.features_by_time
        else:
            data = self._data_loader.filter_multicolor(min_nodes=self._params['min_nodes'])
        RefaelML(self._params, data).run()
        # rml = RefaelML(self._params, data)
        # rml.get_nan_graphs()
        # rml.get_filtered_mx_and_labels()

    def run_al(self, rand=None):
        return self._run_al_bi(rand) if self._params["task"] == "Binary" else self._run_al_multi(rand)

    def run_al_bi_avg(self, rand=None):
        # A function for op. research, averages results from 10 runs.
        data = self._data_loader.filter_by_nodes(min_nodes=self._params['min_nodes']) if self._params['min_nodes'] \
            else self._data_loader.features_by_time
        temp_res = [np.empty(shape=(0, len(data))), np.empty(shape=(0, len(data))),
                    np.empty(shape=(0, len(data))), np.empty(shape=(0, len(data)))]
        temp_perf = [np.empty(shape=(0, len(data))), np.empty(shape=(0, len(data))),
                     np.empty(shape=(0, len(data))), np.empty(shape=(0, len(data)))]
        temp_rand = [np.empty(shape=(0, len(data))), np.empty(shape=(0, len(data))),
                     np.empty(shape=(0, len(data))), np.empty(shape=(0, len(data)))]
        bests = []
        for run in range(10):
            if rand:
                temp_lm = self._params['learn_method']
                temp_eps = self._params['eps']
                self._params['learn_method'] = "rand"
                self._params["eps"] = 0
                rand_time, rand_recall, rand_precision, rand_false_alarm = \
                    TimedActiveLearningBi(data, self._params).run()
                rand_ = [rand_time, rand_recall, rand_precision, rand_false_alarm]
                temp_rand = [np.vstack((temp_rand[i], rand_[i])) for i in range(4)]
                self._params['learn_method'] = temp_lm
                self._params['eps'] = temp_eps
            al = TimedActiveLearningBi(data, self._params)  # , n_black)
            res = al.run()
            temp_res = [np.vstack((temp_res[i], res[i])) for i in range(4)]
            bests.append(al.best_recall_plot())
            all_time, all_recall, all_precision, all_false_alarm = al.performance_on_all_data()
            perf = [all_time, all_recall, all_precision, all_false_alarm]
            temp_perf = [np.vstack((temp_perf[i], perf[i])) for i in range(4)]
        return [list(np.mean(temp_res[i], axis=0)) for i in range(4)], bests[0], \
               [list(np.mean(temp_rand[i], axis=0)) for i in range(4)], \
               [list(np.mean(temp_perf[i], axis=0)) for i in range(4)]

    def _run_al_bi(self, rand=None):
        # Res, Best, Rand and Perf are used for op. research. In order to plot, uncomment the last lines.
        data = self._data_loader.filter_by_nodes(min_nodes=self._params['min_nodes']) if self._params['min_nodes'] \
            else self._data_loader.features_by_time
        # n_black = self._data_loader.count_black()
        if rand:
            temp_lm = self._params['learn_method']
            temp_eps = self._params['eps']
            self._params['learn_method'] = "rand"
            self._params["eps"] = 0
            rand_time, rand_recall, rand_precision, rand_false_alarm = \
                TimedActiveLearningBi(data, self._params).run()
            rand = [rand_time, rand_recall, rand_precision, rand_false_alarm]
            self._params['learn_method'] = temp_lm
            self._params['eps'] = temp_eps
        al = TimedActiveLearningBi(data, self._params)
        res = al.run()
        best = al.best_recall_plot()
        all_time, all_recall, all_precision, all_false_alarm = al.performance_on_all_data()
        perf = [all_time, all_recall, all_precision, all_false_alarm]
        # al.all_data_performance_plot(all_time, all_recall, all_precision, all_false_alarm)
        # al.recall_plot(extra_line=(rand_x, rand_y)) if rand else al.recall_plot()
        # al.precision_plot(extra_line=(rand_x, rand_prec)) if rand else al.precision_plot()
        # al.false_alarm_plot(extra_line=(rand_x, rand_false_alarm)) if rand else al.false_alarm_plot()
        return res, best, rand, perf

    def _run_al_multi(self, rand=None):
        data = self._data_loader.filter_by_nodes(min_nodes=self._params['min_nodes']) if self._params['min_nodes'] \
            else self._data_loader.features_by_time
        if rand:
            temp_lm = self._params['learn_method']
            temp_eps = self._params['eps']
            self._params['learn_method'] = "rand"
            self._params["eps"] = 0
            rand_recalls, rand_acc = TimedActiveLearningMulti(data, self._params).run()
            rand = [rand_recalls, rand_acc]
            self._params['learn_method'] = temp_lm
            self._params['eps'] = temp_eps
        al = TimedActiveLearningMulti(data, self._params)
        recalls, acc = al.run()
        res = [recalls, acc]
        best = al.best_recall_plot()
        al.recall_plot(extra_line=rand_recalls) if rand else al.recall_plot()
        al.accuracy_plot(extra_line=rand_acc) if rand else al.accuracy_plot()
        return res, best, rand


if __name__ == "__main__":

    r = RefaelLearner()
    if ACTIVE_LEARNING:
        r.run_al(rand=True)

    if MACHINE_LEARNING:
        r.run_ml()
    end = 0


