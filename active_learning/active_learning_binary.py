import datetime
import os
from collections import Counter
import numpy as np
from bokeh.plotting import figure, show, save
from bokeh.io import export_png
from smart_selectors import SmartSelector


class TimedActiveLearningBi:
    def __init__(self, data: list, params):  # , n_black):
        self._base_dir = __file__.replace("/", os.sep)
        self._base_dir = os.path.join(self._base_dir.rsplit(os.sep, 1)[0], "..")
        self._params = params
        self._recall = [(0, 0)]
        self._precision = [(0, 0)]  # [(time, precision)]
        self._false_alarm = [(0, 0)]
        self._temp_pred_label = []  # temporal list of [(prediction, label)] that will help calculating precision.
        self._data_by_time = data
        self._test = self._data_by_time[0]
        self._train = {}
        self._init_variables(params)
        self._for_all_data_performance()
        # self._n_black = n_black
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

    def _for_all_data_performance(self):
        self._all_recall = [(0, 0)]
        self._all_precision = [(0, 0)]  # [(time, precision)]
        self._all_false_alarm = [(0, 0)]

    # ---------------------- NOTE call only after run is activated
    def best_recall_plot(self):
        xb = [i for i in range(self._len)]
        yb = [0]
        best = [0]
        blacks_by_time = []
        all_labels = {}
        for t in self._data_by_time:
            for name, (vec, label) in t.items():
                all_labels[name] = label
            blacks_by_time.append(sum([val for key, val in Counter(all_labels.values()).items() if key != self._white]))
        for i in range(1, self._len):
            best.append(best[i-1] + min(self._batch_size*self._queries_per_time, blacks_by_time[i] - best[i-1]))
            yb.append(float(best[i]) / self._n_black)
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
        # 0 < a < eps -> distance based  || at least one black and white revealed -> one_class/ euclidean
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
                self._found[1 - self._white] += 1
            else:
                self._found[self._white] += 1

            # add feature vec to train
            self._train[name] = self._test[name]
            del self._test[name]

    def _performance_on_all_data(self, time):
        # Measures performance on all test data we have every time.
        # Recall, precision and false alarm are measured relative to the whole data at every time, not to the whole data
        # up until that time.
        performance_on_all = SmartSelector(self._params, batch_size=np.inf, ml_method=self._params['learn_method'])
        if len(self._test) <= 0:
            self._all_recall.append((time, 0))
            self._all_precision.append((time, 0))
            self._all_false_alarm.append((time, 0))
            return
        name_pred_label = performance_on_all.ml_select(self._train, self._test)
        tp = sum([1 if round(t[1]) == t[2] else 0 for t in name_pred_label if round(t[1]) != self._white])
        fp = sum([0 if round(t[1]) == t[2] else 1 for t in name_pred_label if round(t[1]) != self._white])
        fn = sum([0 if round(t[1]) == t[2] else 1 for t in name_pred_label if round(t[1]) == self._white])
        self._all_recall.append((time, (tp / (tp + fn) if tp + fn else 0)))
        self._all_precision.append((time, (tp / (tp + fp) if (tp + fp) else 0)))
        self._all_false_alarm.append((time, fp))

    def run(self):
        # number of queries made AFTER first exploration
        queries_made = 2 if self._batch_size == 1 else 1
        start_time = 1
        # forward until there is some graph to start from ( first times may not contain any graph )
        flag = 0
        while len(self._test) == 0:
            self._forward_time()
            flag += 1
            self._recall.append((start_time, self._found[1 - self._white] / self._n_black))
            self._precision.append((start_time, 0))  # By this time, no guesses were made.
            self._false_alarm.append((start_time, 0))  # No wrong guesses were made as well.
            self._all_recall.append((start_time, 0))
            self._all_precision.append((start_time, 0))
            self._all_false_alarm.append((start_time, 0))
        self._first_exploration()

        # first exploration - reveal by distance
        for i in range(start_time + flag, self._len):
            print("-----------------------------------    TIME " + str(i) + "    -------------------------------------")
            # as long as number as queries made < K  &&  test contain some data to ask about.
            while queries_made < self._queries_per_time and len(self._test) >= self._batch_size:
                self._explore_exploit()
                queries_made += 1
            queries_made = 0
            # print results for current time, then forward time
            self._performance_on_all_data(i)
            print("test_len =" + str(len(self._test)) + ", train len=" + str(len(self._train)) + ", total =" +
                  str(len(self._train) + len(self._test)))
            temp_to_prec = [1 if round(t[0]) == t[1] else 0 for t in self._temp_pred_label if
                            round(t[0]) != self._white]
            false_alarm = [1 if round(t[0]) != t[1] and t[1] == self._white else 0 for t in self._temp_pred_label]
            self._precision.append((i, (sum(temp_to_prec) / len(temp_to_prec) if len(temp_to_prec) else 0)))
            self._recall.append((i, self._found[1 - self._white] / self._n_black))
            self._false_alarm.append((i, sum(false_alarm)))
            print(str(self._found[1 - self._white]) + " / " + str(self._n_black))
            self._forward_time()
        return [x for x, y in self._recall], [y for x, y in self._recall], [p[1] for p in self._precision], \
            [f[1] for f in self._false_alarm]

    def performance_on_all_data(self):
        return [x for x, y in self._all_recall], [y for x, y in self._all_recall], \
               [p[1] for p in self._all_precision], [f[1] for f in self._all_false_alarm]

    def recall_plot(self, extra_line=None):
        g_title = "AL recall over time - " + self._params['learn_method'] + " - window: " + str(self._params['window_size'])
        p = figure(plot_width=600, plot_height=250, title=g_title,
                   x_axis_label="time", y_axis_label="recall")
        if extra_line:
            p.line(extra_line[0], extra_line[1], line_color='red')
        p.line([x for x, y in self._recall], [y for x, y in self._recall], line_color='blue')
        best_x, best_y = self.best_recall_plot()
        p.line(best_x, best_y, line_color='green')
        plot_name = "AL_recall_" + datetime.datetime.now().strftime("%d%m%y_%H%M%S")
        # save(p, os.path.join(self._base_dir, "fig", "active_learning", plot_name + ".html"))
        export_png(p, os.path.join(self._base_dir, "fig", "active_learning", plot_name + ".png"))
        param_file = open(os.path.join(self._base_dir, "fig", "active_learning", plot_name + "_params.txt"), "wt")
        param_file.write(str(self._params))
        param_file.close()

    def false_alarm_plot(self, extra_line=None):
        g_title = "AL false alarm over time - " + self._params['learn_method'] + " - window: " + \
                  str(self._params['window_size'])
        p = figure(plot_width=600, plot_height=250, title=g_title,
                   x_axis_label="time", y_axis_label="false alarm")
        if extra_line:
            p.line(extra_line[0], extra_line[1], line_color='red')
        p.line([x for x, y in self._false_alarm], [y for x, y in self._false_alarm], line_color='blue')
        plot_name = "AL_false_alarm_" + datetime.datetime.now().strftime("%d%m%y_%H%M%S")
        # save(p, os.path.join(self._base_dir, "fig", "active_learning", plot_name + ".html"))
        export_png(p, os.path.join(self._base_dir, "fig", "active_learning", plot_name + ".png"))
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
        # save(p, os.path.join(self._base_dir, "fig", "active_learning", plot_name + ".html"))
        export_png(p, os.path.join(self._base_dir, "fig", "active_learning", plot_name + ".png"))
        param_file = open(os.path.join(self._base_dir, "fig", "active_learning", plot_name + "_params.txt"), "wt")
        param_file.write(str(self._params))
        param_file.close()

    def all_data_performance_plot(self, time, recall, precision, false_alarm):
        g_titles = ["False alarm of all data by time, given AL training",
                    "Recall of all data by time, given AL training", "Precision of all data by time, given AL training"]
        plot_names = ["all_data_false_alarm_" + datetime.datetime.now().strftime("%d%m%y_%H%M%S"),
                      "all_data_recall_" + datetime.datetime.now().strftime("%d%m%y_%H%M%S"),
                      "all_data_precision_" + datetime.datetime.now().strftime("%d%m%y_%H%M%S")]
        p = figure(plot_width=600, plot_height=250, title=g_titles[0],
                   x_axis_label="time", y_axis_label="false alarm")
        p.circle(time, false_alarm, size=5, color='blue', alpha=0.5)
        export_png(p, os.path.join(self._base_dir, "fig", "al_on_all_data", plot_names[0] + ".png"))
        param_file_p = open(os.path.join(self._base_dir, "fig", "al_on_all_data", plot_names[0] + "_params.txt"), "wt")
        param_file_p.write(str(self._params))
        param_file_p.close()

        q = figure(plot_width=600, plot_height=250, title=g_titles[1],
                   x_axis_label="time", y_axis_label="recall")
        q.circle(time, recall, size=5, color='blue', alpha=0.5)
        export_png(q, os.path.join(self._base_dir, "fig", "al_on_all_data", plot_names[1] + ".png"))
        param_file_q = open(os.path.join(self._base_dir, "fig", "al_on_all_data", plot_names[1] + "_params.txt"), "wt")
        param_file_q.write(str(self._params))
        param_file_q.close()

        r = figure(plot_width=600, plot_height=250, title=g_titles[2],
                   x_axis_label="time", y_axis_label="precision")
        r.circle(time, precision, size=5, color='blue', alpha=0.5)
        export_png(r, os.path.join(self._base_dir, "fig", "al_on_all_data", plot_names[2] + ".png"))
        param_file_r = open(os.path.join(self._base_dir, "fig", "al_on_all_data", plot_names[2] + "_params.txt"), "wt")
        param_file_r.write(str(self._params))
        param_file_r.close()
