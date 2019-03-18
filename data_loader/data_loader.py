import pickle
from collections import Counter
import os
from features_meta import NODE_FEATURES_ML
from features_processor import FeaturesProcessor  # , log_norm
from graph_features import GraphFeatures
from loggers import PrintLogger
# from motif_correlation.beta_calculator import LinearContext
# from motif_correlation.features_picker import PearsonFeaturePicker
from temporal_multi_graph import TemporalMultiGraph
# import numpy as np


class DataLoader:
    def __init__(self, params):
        self._base_dir = __file__.replace("/", os.sep)
        self._base_dir = os.path.join(self._base_dir.rsplit(os.sep, 1)[0], "..")
        self._params = params
        self._features_by_time = []
        self._multi_graphs_by_time = []
        self._database = None
        self._num_blacks = None
        self._calc_features()

    @property
    def features_by_time(self):
        return self._features_by_time

    @property
    def multi_graphs_by_time(self):
        return self._multi_graphs_by_time

    def _ftr_pkl_name(self):
            return self._params['database_full_name'] + ".pkl"
            # return self._params['database'] + "__ds_" + str(self._params['days_split']) + "_st_" + \
            #        str(self._params['start_time']) + "_ws_" + str(self._params['window_size']) + "_d_" + \
            #        str(self._params['directed']) + "_mc_" + str(self._params['max_connected'] + ".pkl")

    def _load_database(self, force_build=False):
        if self._database and not force_build:
            return
        self._database = TemporalMultiGraph(self._params['database_full_name'],
                                            os.path.join(self._base_dir,
                                                         'INPUT_DATABASE', self._params['data_file_name']),
                                            time_format='MIL',
                                            time_col='StartTime',
                                            src_col='SourceID',
                                            dst_col='DestinationID',
                                            label_col='target',
                                            subgraph_name_col='Community',
                                            days=self._params['days_split'],
                                            time_format_out=self._params['date_format'],
                                            directed=self._params['directed'])

    def count_black(self):
        if not self._num_blacks:
            self._load_database()
            print(Counter(self._database.labels.values()))
            self._num_blacks = sum([val for key, val in Counter(self._database.labels.values()).items() if
                                    key != self._params['white_label']])
        return self._num_blacks

    def _calc_features(self, pkl=True):
        # load dictionary if exists
        if pkl and self._ftr_pkl_name() in os.listdir(
                os.path.join(self._base_dir, 'pkl', 'ftr_by_time_dictionaries')):
            self._features_by_time, self._multi_graphs_by_time = \
                pickle.load(open(os.path.join(self._base_dir, 'pkl', 'ftr_by_time_dictionaries',
                                              self._ftr_pkl_name()), "rb"))
            return

        self._load_database()
        labels = self._database.labels
        # make directory for database
        dir_path = os.path.join(self._base_dir, 'pkl', 'graph_measures')
        if self._params['database_full_name'] not in os.listdir(dir_path):
            os.mkdir(os.path.join(dir_path, self._params['database_full_name']))
        dir_path = os.path.join(dir_path, self._params['database_full_name'])

        # calculate features
        for i, multi_graph in enumerate(self._database.multi_graph_by_window(self._params['window_size'],
                                        self._params['start_time'])):
            if "time_" + str(i) not in os.listdir(dir_path):
                os.mkdir(os.path.join(dir_path, "time_" + str(i)))
            mg_dir_path = os.path.join(dir_path, "time_" + str(i))

            ftr_tmp_dict = {}
            # nodes_and_edges = {}
            for name in multi_graph.graph_names():
                if name not in os.listdir(mg_dir_path):
                    os.mkdir(os.path.join(mg_dir_path, name))
                gnx_dir_path = os.path.join(mg_dir_path, name)

                raw_ftr = GraphFeatures(multi_graph.get_gnx(name), NODE_FEATURES_ML, dir_path=gnx_dir_path,
                                        is_max_connected=self._params['max_connected'],
                                        logger=PrintLogger(self._params['database_full_name']))
                raw_ftr.build(should_dump=True)  # build features
                nodes_and_edges = [multi_graph.node_count(graph_id=name), multi_graph.edge_count(graph_id=name)]
                # nodes_and_edges[name] = [multi_graph.node_count(graph_id=name), multi_graph.edge_count(graph_id=name)]

                # ====================== motif ratio ========================
                ftr_tmp_dict[name] = (FeaturesProcessor(raw_ftr).activate_motif_ratio_vec(to_add=nodes_and_edges),
                                      labels[name])

                # ==================== ftr correlation ======================
                # ftr_tmp_dict[name] = (FeaturesProcessor(raw_ftr).as_matrix(norm_func=log_norm))
                # ftr_tmp_dict[name] = (FeaturesProcessor(raw_ftr).as_matrix())

            # concat_mx = np.vstack([mx for name, mx in ftr_tmp_dict.items()])
            # pearson_picker = PearsonFeaturePicker(concat_mx, size=self._params['ftr_pairs'],
            #                                       identical_bar=0.9)
            # best_pairs = pearson_picker.best_pairs()
            # beta = LinearContext(multi_graph, ftr_tmp_dict, best_pairs, window_size=len(ftr_tmp_dict))
            # beta_matrix = beta.beta_matrix()
            # node and edges can pe appended here
            # for j, name in enumerate(multi_graph.graph_names()):
            #     ftr_tmp_dict[name] = (np.hstack((beta_matrix[j], nodes_and_edges[name])), labels[name])

            self._features_by_time.append(ftr_tmp_dict)

            multi_graph.suspend_logger()
            self._multi_graphs_by_time.append(multi_graph)

        pickle.dump((self._features_by_time, self._multi_graphs_by_time),
                    open(os.path.join(self._base_dir, 'pkl', 'ftr_by_time_dictionaries', self._ftr_pkl_name()), "wb"))

    def filter_by_nodes(self, min_nodes=5):
        filtered = []
        for i in range(len(self._multi_graphs_by_time)):
            print("filter time " + str(i))
            self._multi_graphs_by_time[i].filter(
                lambda x: False if self._multi_graphs_by_time[i].node_count(x) < min_nodes else True,
                func_input="graph_name")
            ftr_tmp_dict = {}
            for name in self._multi_graphs_by_time[i].graph_names():
                ftr_tmp_dict[name] = self._features_by_time[i][name]
            filtered.append(ftr_tmp_dict)
        return filtered

    def filter_multicolor(self, min_nodes=5):
        # Take care of labels of multicolored data
        if self._params['task'] == 'Multiclass1':
            filtered = self._filter_multi1(min_nodes)
        elif self._params['task'] == 'Multiclass2':
            filtered = self._filter_multi2(min_nodes)
        else:
            raise ValueError("Multicolored data was chosen, but the parameter 'task' is 'Binary'")
        return filtered

    def _filter_multi1(self, min_nodes=5):
        # Currently, class 2 (the very small one) is still interesting and theoretically we can learn about it.
        filtered = []
        for i in range(len(self._multi_graphs_by_time)):
            print("filter time " + str(i))
            self._multi_graphs_by_time[i].filter(
                lambda x: False if self._multi_graphs_by_time[i].node_count(x) < min_nodes else True,
                func_input="graph_name")
            ftr_tmp_dict = {}
            for name in self._multi_graphs_by_time[i].graph_names():
                if self._features_by_time[i][name][1] == 3:
                    ftr_tmp_dict[name] = (self._features_by_time[i][name][0], 0)
                else:
                    ftr_tmp_dict[name] = (self._features_by_time[i][name][0], self._features_by_time[i][name][1])
            filtered.append(ftr_tmp_dict)
        return filtered

    def _filter_multi2(self, min_nodes):
        filtered = []
        for i in range(len(self._multi_graphs_by_time)):
            print("filter time " + str(i))
            self._multi_graphs_by_time[i].filter(
                lambda x: False if (
                            self._features_by_time[i][x][1] == 6 or self._multi_graphs_by_time[i].node_count(x) <
                            min_nodes) else True, func_input="graph_name")
            ftr_tmp_dict = {}
            for name in self._multi_graphs_by_time[i].graph_names():
                if self._features_by_time[i][name][1] == 1:
                    ftr_tmp_dict[name] = (self._features_by_time[i][name][0], 0)
                elif 2 <= self._features_by_time[i][name][1] <= 5:
                    ftr_tmp_dict[name] = (self._features_by_time[i][name][0], 1)
                elif self._features_by_time[i][name][1] > 6:
                    ftr_tmp_dict[name] = (self._features_by_time[i][name][0], self._features_by_time[i][name][1] - 5)
            filtered.append(ftr_tmp_dict)
        return filtered
