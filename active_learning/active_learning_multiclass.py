import datetime
import os
from collections import Counter
from operator import itemgetter

import numpy as np
from bokeh.plotting import figure, show, save
from bokeh.io import export_png
from smart_selectors import SmartSelector


class TimedActiveLearningMulti:
    def __init__(self, data: list, params):
        self._base_dir = __file__.replace("/", os.sep)
        self._base_dir = os.path.join(self._base_dir.rsplit(os.sep, 1)[0], "..")
        self._params = params
        self._accuracy = [(0, 0)]
        self._temp_pred_label = []  # temporal list of [(prediction, label)] that will help calculating accuracy.
        self._data_by_time = data
        self._test = self._data_by_time[0]
        self._train = {}
        self._init_variables(params)

    def _init_variables(self, params):
        self._num_classes = 4 if self._params['task'] == "Multiclass1" else 8
        self._batch_size = params['batch_size']
        self._eps = params['eps']
        self._target_recall = params['target_recall']
        self._queries_per_time = params['queries_per_time']
        self._selector = SmartSelector(params, self._batch_size, ml_method=params['learn_method'],
                                       distance_method="euclidean", num_classes=self._num_classes)
        self._n_color, self._total_communities = self._count_color()  # number of every color
        self._recalls = [[]] * self._num_classes
        self._stop_cond = np.round(self._target_recall * self._total_communities)  # number to find - stop condition
        self._time = 0  # how many nodes we asked about
        self._found = [0] * self._num_classes
        self._first_time = True
        self._len = len(self._data_by_time)

    # ---------------------- NOTE call only after run is activated
    def best_recall_plot(self):
        plot_by_colors = []
        for color in range(self._num_classes):
            color_count_dict = {}
            total_count_dict = {}
            color_over_time = [0]
            total_over_time = [0]
            for t in self._data_by_time:
                for name, (vec, label) in t.items():
                    total_count_dict[name] = label
                    if label == color:
                        color_count_dict[name] = label
                color_over_time.append(len(color_count_dict))
                total_over_time.append(len(total_count_dict))

            queries = [0] + [self._queries_per_time * self._batch_size * (i+1) for i in range(self._len)]

            xb = [i for i in range(self._len)]
            yb = []
            guessed = [0]
            for i in range(self._len + 1):
                if queries[i] - guessed[i-1] < self._queries_per_time * self._batch_size:
                    guessed.append(guessed[i-1])
                    continue
                guessed.append(min(total_over_time[i], queries[i]))
                best_color = min(color_over_time[i], queries[i])

                yb.append(best_color / self._n_color[color])
            plot_by_colors.append((xb, yb))
        return plot_by_colors

    def _count_color(self):
        all_labels = {}
        for t in self._data_by_time:
            for name, (vec, label) in t.items():
                all_labels[name] = label
        labels = Counter(all_labels.values())
        print(labels)
        color_list = [population for cl, population in sorted(labels.items(), key=itemgetter(0))]
        return color_list, sum(color_list)

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
            self._found[int(self._test[name][1])] += 1
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
        for cl in range(self._num_classes):
            self._recalls[cl] = self._recalls[cl] + [(start_time - 1, self._found[cl] / self._n_color[cl])]
        self._accuracy.append((start_time - 1, 0))  # By this time, no guesses were made.
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
            temp_to_prec = [1.0 / self._n_color[int(t[1])] if round(t[0]) == t[1] else 0 for t in self._temp_pred_label]

            self._accuracy.append((i, (sum(temp_to_prec)/self._num_classes if len(temp_to_prec) else 0)))
            for cl in range(self._num_classes):
                self._recalls[cl].append((i, self._found[cl] / self._n_color[cl]))
                print("Class " + str(cl) + ": " + str(self._found[cl]) + " / " + str(self._n_color[cl]))
            print("Accuracy: " + str(sum(temp_to_prec) / self._num_classes))
            self._forward_time()
        return self._recalls, self._accuracy

    def recall_plot(self, extra_line=None):
        best_plot = self.best_recall_plot()
        for col in range(self._num_classes):
            g_title = "AL recall over time - " + str(col) + " - " + self._params['learn_method'] + " - window: " + \
                      str(self._params['window_size'])
            p = figure(plot_width=600, plot_height=250, title=g_title,
                       x_axis_label="time", y_axis_label="recall")
            if extra_line:
                extra_time = [extra_line[col][t][0] for t in range(len(extra_line[col]))]
                extra_recall = [extra_line[col][t][1] for t in range(len(extra_line[col]))]
                p.line(extra_time, extra_recall, line_color='red')
            time = [self._recalls[col][t][0] for t in range(len(self._recalls[col]))]
            rec = [self._recalls[col][t][1] for t in range(len(self._recalls[col]))]
            p.line(time, rec, line_color='blue')
            best_x, best_y = best_plot[col]
            p.line(best_x, best_y, line_color='green')
            plot_name = "AL_recall_" + str(col) + "_" + datetime.datetime.now().strftime("%d%m%y_%H%M%S")
            # save(p, os.path.join(self._base_dir, "fig", "active_learning", plot_name + ".html"))
            export_png(p, os.path.join(self._base_dir, "fig", "active_learning", plot_name + ".png"))
            param_file = open(os.path.join(self._base_dir, "fig", "active_learning", plot_name + "_params.txt"), "wt")
            param_file.write(str(self._params))
            param_file.close()

    def accuracy_plot(self, extra_line=None):
        graph_title = "AL accuracy over time - " + self._params['learn_method'] + " - window: " \
                      + str(self._params['window_size'])
        p = figure(plot_width=600, plot_height=250, title=graph_title,
                   x_axis_label="time", y_axis_label="weighted accuracy")
        if extra_line:
            extra_time = [extra_line[t][0] for t in range(len(extra_line))]
            extra_acc = [extra_line[t][1] for t in range(len(extra_line))]
            p.line(extra_time, extra_acc, line_color='red')
        p.line([p[0] for p in self._accuracy], [p[1] for p in self._accuracy], line_color='blue')
        plot_name = "AL_accuracy_" + datetime.datetime.now().strftime("%d%m%y_%H%M%S")
        # save(p, os.path.join(self._base_dir, "fig", "active_learning", plot_name + ".html"))
        export_png(p, os.path.join(self._base_dir, "fig", "active_learning", plot_name + ".png"))
        param_file = open(os.path.join(self._base_dir, "fig", "active_learning", plot_name + "_params.txt"), "wt")
        param_file.write(str(self._params))
        param_file.close()
