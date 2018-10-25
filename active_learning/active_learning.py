import datetime
import os
from collections import Counter
import numpy as np
from active_learning.smart_selctors import SmartSelector
from bokeh.plotting import figure, show, save

BLACK = 0
OTHER = 1


class TimedActiveLearning:
    def __init__(self, data: list, params):
        self._base_dir = __file__.replace("/", os.sep)
        self._base_dir = os.path.join(self._base_dir.rsplit(os.sep, 1)[0], "..")
        self._params = params
        self._recall = []
        self._data_by_time = data
        self._test = self._data_by_time[0]
        self._train = {}
        self._init_variables(params)
        self._n_black = self._count_black(params)                                        # number of blacks
        self._stop_cond = np.round(self._target_recall * self._n_black)  # number of blacks to find - stop condition

    def _init_variables(self, params):
        self._white = params['white_label']                              # who is the _white
        self._batch_size = params['batch_size']
        self._eps = params['eps']
        self._target_recall = params['target_recall']
        self._queries_per_time = params['queries_per_time']
        self._selector = SmartSelector(params, self._batch_size, ml_method=params['ml_method'],
                                       distance_method="euclidean")

        self._time = 0  # how many nodes we asked about
        self._found = [0, 0]
        self._first_time = True
        self._len = len(self._data_by_time)

    def _count_black(self, params):
        all_labels = {}
        for t in self._data_by_time:
            for name, (vec, label) in t.items():
                all_labels[name] = label
        print(Counter(all_labels.values()))
        return sum([val for key, val in Counter(all_labels.values()).items() if key != params['white_label']])

    def _forward_time(self):
        self._time += 1
        for name, vec_and_label in self._data_by_time[self._time].items():
            if name in self._train:
                self._train[name] = vec_and_label
            else:
                self._test[name] = vec_and_label

    def _first_exploration(self):
        # explore first using distance
        start_k = 2 if self._batch_size == 1 else 1
        for i in range(start_k):
            # first two nodes (index)
            top_index = self._selector.distance_select(self._test, self._test)
            self._reveal(top_index)

    def _explore_exploit(self):
        rand = np.random.uniform(0, 1)
        # 0 < a < eps -> distance based  || at least one black and white reviled -> one_class/ euclidean
        if rand < self._eps or min(self._found) == 0:
            # idx -> most far away node index
            top = self._selector.distance_select(self._train, self._test)
        else:
            # idx -> by learning
            top = self._selector.ml_select(self._train, self._test)
        self._reveal(top)

    def _reveal(self, top):
        for name in top:
            if self._test[name][1] != self._white:
                self._found[BLACK] += 1
            else:
                self._found[OTHER] += 1

            # add feature vec to train
            self._train[name] = self._test[name]
            del self._test[name]

    def run(self):
        queries_made = 2 if self._batch_size == 1 else 1
        self._first_exploration()
        for i in range(self._len - 1):      # TODO stop when target recall reached
            print("-----------------------------------    TIME " + str(i) + "    -------------------------------------")
            while queries_made <= self._queries_per_time and len(self._test) >= self._queries_per_time:
                self._explore_exploit()
                queries_made += 1
            queries_made = 0
            print("test_len =" + str(len(self._test)) + ", train len=" + str(len(self._train)) + ", total =" +
                  str(len(self._train) + len(self._test)))
            self._forward_time()
            self._recall.append(self._found[BLACK] / self._n_black)
            print(str(self._found[BLACK]) + " / " + str(self._n_black))

    def plot(self):
        p = figure(plot_width=600, plot_height=250, title="AL recall over time - " + self._params['ml_method'],
                   x_axis_label="revealed:  (time*batch_size)/total_communities", y_axis_label="recall")
        p.line(list(range(self._len)), [y / self._len for y in range(self._len)], line_color='red')
        p.line(list(range(len(self._recall))), self._recall, line_color='blue')
        plot_name = "AL_" + datetime.datetime.now().strftime("%d%m%y_%H%M%S")
        save(p, os.path.join(self._base_dir, "fig", "active_learning", plot_name + ".html"))
        param_file = open(os.path.join(self._base_dir, "fig", "active_learning", plot_name + "_params.txt"), "wt")
        param_file.write(str(self._params))
        param_file.close()
