import datetime
import os
from collections import Counter
import numpy as np
from bokeh.plotting import figure, show, save
from smart_selectors import SmartSelector

BLACK = 0
OTHER = 1


class TimedActiveLearningBi:
    def __init__(self, data: list, params):
        self._base_dir = __file__.replace("/", os.sep)
        self._base_dir = os.path.join(self._base_dir.rsplit(os.sep, 1)[0], "..")
        self._params = params
        self._recall = [(0, 0)]
        self._precision = [(0, 0)]  # [(time, precision)]
        self._temp_pred_label = []  # temporal list of [(prediction, label)] that will help calculating precision.
        self._data_by_time = data
        self._test = self._data_by_time[0]
        self._train = {}
        self._init_variables(params)
        self._n_black, self._total_communities = self._count_black(params)                        # number of blacks
        self._stop_cond = np.round(self._target_recall * self._n_black)  # number of blacks to find - stop condition

    def _init_variables(self, params):
        self._white = params['white_label']
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
        black_count_dict = {}
        total_count_dict = {}
        blacks_over_time = [0]
        total_over_time = [0]
        for t in self._data_by_time:
            for name, (vec, label) in t.items():
                total_count_dict[name] = label
                if label != self._white:
                    black_count_dict[name] = label
            blacks_over_time.append(len(black_count_dict))
            total_over_time.append(len(total_count_dict))

        queries = [0] + [self._queries_per_time * self._batch_size * (i+1) for i in range(self._len)]

        xb = []
        yb = []
        guessed = [0]
        for i in range(self._len + 1):
            if queries[i] - guessed[i-1] < self._queries_per_time * self._batch_size:
                guessed.append(guessed[i-1])
                continue
            guessed.append(min(total_over_time[i], queries[i]))
            best_black = min(blacks_over_time[i], queries[i])

            xb.append(guessed[i] / self._total_communities)
            yb.append(best_black / self._n_black)
        return xb, yb

    def _count_black(self, params):
        all_labels = {}
        for t in self._data_by_time:
            for name, (vec, label) in t.items():
                all_labels[name] = label
        print(Counter(all_labels.values()))
        return sum([val for key, val in Counter(all_labels.values()).items() if key != params['white_label']]), \
               len(all_labels)

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
            name_pred_label = self._selector.ml_select(self._train, self._test)
            top = []
            for t in name_pred_label:
                top.append(t[0])
                self._temp_pred_label.append((t[1], t[2]))
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
        # number of queries made AFTER first exploration
        queries_made = 2 if self._batch_size == 1 else 1
        start_time = 1
        # forward until there is some graph to start from ( first times may not contain any graph )
        while len(self._test) == 0:
            self._forward_time()
            start_time += 1

        # first exploration - reveal by distance
        self._first_exploration()
        self._recall.append((len(self._train) / self._total_communities, self._found[BLACK] / self._n_black))
        self._precision.append((len(self._train) / self._total_communities, 0))  # By this time, no guesses were made.
        for i in range(start_time, self._len):
            print("-----------------------------------    TIME " + str(i) + "    -------------------------------------")
            # self._temp_pred_label = []  # If precision(t) = True_guesses/total_guesses (time<=t), ctrl+/
            # as long as number as queries made < K  &&  test contain some data to ask about
            while queries_made < self._queries_per_time and len(self._test) >= self._batch_size:
                self._explore_exploit()
                queries_made += 1
            queries_made = 0
            # print results for current time + forward time
            print("test_len =" + str(len(self._test)) + ", train len=" + str(len(self._train)) + ", total =" +
                  str(len(self._train) + len(self._test)))
            temp_to_prec = [1 if round(t[0]) == t[1] else 0 for t in self._temp_pred_label if round(t[0]) != self._white]

            self._precision.append((len(self._train) / self._total_communities,
                                    (sum(temp_to_prec)/len(temp_to_prec) if len(temp_to_prec) else 0)))
            self._recall.append((len(self._train) / self._total_communities, self._found[BLACK] / self._n_black))
            print(str(self._found[BLACK]) + " / " + str(self._n_black))
            self._forward_time()
        return [x for x, y in self._recall], [y for x, y in self._recall], [p[1] for p in self._precision]

    def recall_plot(self, extra_line=None):
        g_title = "AL recall over time - " + self._params['learn_method'] + " - window: " + str(self._params['window_size'])
        p = figure(plot_width=600, plot_height=250, title=g_title,
                   x_axis_label="revealed:  (time*batch_size)/total_communities", y_axis_label="recall")
        if extra_line:
            p.line(extra_line[0], extra_line[1], line_color='red')
        p.line([x for x, y in self._recall], [y for x, y in self._recall], line_color='blue')
        best_x, best_y = self.best_recall_plot()
        p.line(best_x, best_y, line_color='green')
        plot_name = "AL_recall_" + datetime.datetime.now().strftime("%d%m%y_%H%M%S")
        save(p, os.path.join(self._base_dir, "fig", "active_learning", plot_name + ".html"))
        param_file = open(os.path.join(self._base_dir, "fig", "active_learning", plot_name + "_params.txt"), "wt")
        param_file.write(str(self._params))
        param_file.close()

    def precision_plot(self, extra_line=None):
        graph_title = "AL precision over time - " + self._params['learn_method'] + " - window: " \
                      + str(self._params['window_size'])
        p = figure(plot_width=600, plot_height=250, title=graph_title,
                   x_axis_label="time", y_axis_label="precision")
        if extra_line:
            p.line(extra_line[0], extra_line[1], line_color='red')
        p.line([p[0] for p in self._precision], [p[1] for p in self._precision], line_color='blue')
        plot_name = "AL_precision_" + datetime.datetime.now().strftime("%d%m%y_%H%M%S")
        save(p, os.path.join(self._base_dir, "fig", "active_learning", plot_name + ".html"))
        param_file = open(os.path.join(self._base_dir, "fig", "active_learning", plot_name + "_params.txt"), "wt")
        param_file.write(str(self._params))
        param_file.close()
