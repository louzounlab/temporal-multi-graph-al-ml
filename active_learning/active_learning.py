import datetime
import os
from collections import Counter
import numpy as np
from bokeh.plotting import figure, show, save
from smart_selctors import SmartSelector

BLACK = 0
OTHER = 1


class TimedActiveLearning:
    def __init__(self, data: list, params):
        self._base_dir = __file__.replace("/", os.sep)
        self._base_dir = os.path.join(self._base_dir.rsplit(os.sep, 1)[0], "..")
        self._params = params
        self._recall = [0]
        self._data_by_time = data
        self._test = self._data_by_time[0]
        self._train = {}
        self._init_variables(params)
        self._n_black = self._count_black(params)                        # number of blacks
        self._stop_cond = np.round(self._target_recall * self._n_black)  # number of blacks to find - stop condition

    def _init_variables(self, params):
        self._white = params['white_label']                              # who is the _white
        self._batch_size = params['batch_size']
        self._eps = params['eps']
        self._target_recall = params['target_recall']
        self._queries_per_time = params['queries_per_time']
        self._selector = SmartSelector(params, self._batch_size, ml_method=params['learn_method'],
                                       distance_method="euclidean")
        self._time = 0  # how many nodes we asked about
        self._found = [0, 0]
        self._first_time = True
        self._len = len(self._data_by_time)

    # ---------------------- NOTE call only after run is activated
    def best_recall_plot(self):
        count_dict = {}
        black_time = []
        for t in self._data_by_time:
            for name, (vec, label) in t.items():
                if label != self._white:
                    count_dict[name] = label
            black_time.append(len(count_dict))

        xb = [i for i in range(self._len + 1)]
        yb = [0] + [i / self._n_black for i in black_time]
        return xb, yb

    def _total_num_examples(self):
        examples = []
        for t in self._data_by_time:
            for name, (vec, label) in t.items():
                examples.append(name)
        return len(set(examples))

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
        e = 1

    def run(self):
        # number of queries made AFTER first exploration
        queries_made = 2 if self._batch_size == 1 else 1
        start_time = 0
        # forward until there is some graph to start from ( first times may not contain any graph )
        while len(self._test) == 0:
            self._forward_time()
            start_time += 1
            self._recall.append(0)
        # first exploration - reveal by distance
        self._first_exploration()
        self._recall.append(self._found[BLACK] / self._n_black)
        for i in range(start_time, self._len - 1):      # TODO stop when target recall reached
            print("-----------------------------------    TIME " + str(i) + "    -------------------------------------")
            # as long as number as queries made < K  &&  test contain some data to ask about
            while queries_made < self._queries_per_time and len(self._test) >= self._batch_size:
                self._explore_exploit()
                queries_made += 1
            queries_made = 0
            # print results for current time + forward time
            print("test_len =" + str(len(self._test)) + ", train len=" + str(len(self._train)) + ", total =" +
                  str(len(self._train) + len(self._test)))
            self._recall.append(self._found[BLACK] / self._n_black)
            print(str(self._found[BLACK]) + " / " + str(self._n_black))
            self._forward_time()
        return [i for i in range(self._len + 1)], self._recall

    def plot(self):
        g_title = "AL recall over time - " + self._params['learn_method'] + " - window: " + str(self._params['window_size'])
        p = figure(plot_width=600, plot_height=250, title=g_title,
                   x_axis_label="revealed:  (time*batch_size)/total_communities", y_axis_label="recall")
        x_axis = [(i * self._batch_size * self._queries_per_time) / (len(self._train) + len(self._test))
                  for i in range(self._len)]
        p.line(x_axis, [y / self._len for y in range(self._len)], line_color='red')
        p.line(x_axis, self._recall, line_color='blue')
        plot_name = "AL_" + datetime.datetime.now().strftime("%d%m%y_%H%M%S")
        save(p, os.path.join(self._base_dir, "fig", "active_learning", plot_name + ".html"))
        param_file = open(os.path.join(self._base_dir, "fig", "active_learning", plot_name + "_params.txt"), "wt")
        param_file.write(str(self._params))
        param_file.close()
