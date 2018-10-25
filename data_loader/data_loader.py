import pickle
from collections import Counter
import os
from features_meta import NODE_FEATURES_ML
from features_processor import FeaturesProcessor
from graph_features import GraphFeatures
from loggers import PrintLogger
from temporal_multi_graph import TemporalMultiGraph


class DataLoader:
    def __init__(self, params):
        self._base_dir = __file__.replace("/", os.sep)
        self._base_dir = os.path.join(self._base_dir.rsplit(os.sep, 1)[0])
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
        return self._params['database'] + "_" + str(self._params['days_split']) + "_" + str(
            self._params['start_time']) \
               + "_" + str(self._params['window_size']) + "_" + str(self._params['directed']) + "_" + \
               str(self._params['max_connected']) + ".pkl"

    def _load_database(self, force_build=False):
        if self._database and not force_build:
            return
        self._database = TemporalMultiGraph(self._params['database'],
                                            os.path.join('INPUT_DATABASE', self._params['data_file_name']),
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
                os.path.join(self._base_dir, "..", 'pkl', 'ftr_by_time_dictionaries')):
            self._features_by_time, self._multi_graphs_by_time = \
                pickle.load(open(os.path.join(self._base_dir, "..", 'pkl', 'ftr_by_time_dictionaries',
                                              self._ftr_pkl_name()), "rb"))
            return

        self._load_database()
        labels = self._database.labels
        # make directory for database
        dir_path = os.path.join(self._base_dir, 'pkl', 'graph_measures', self._params['database'])
        if self._params['database'] not in os.listdir(os.path.join(self._base_dir, 'pkl', 'graph_measures')):
            os.mkdir(dir_path)

        # calculate features
        for multi_graph in self._database.multi_graph_by_window(self._params['window_size'],
                                                                self._params['start_time']):
            ftr_tmp_dict = {}
            for name in multi_graph.graph_names():
                raw_ftr = GraphFeatures(multi_graph.get_gnx(name), NODE_FEATURES_ML, dir_path,
                                        is_max_connected=self._params['max_connected'],
                                        logger=PrintLogger(self._params['database']))
                nodes_and_edges = [multi_graph.node_count(graph_id=name), multi_graph.edge_count(graph_id=name)]
                ftr_tmp_dict[name] = (FeaturesProcessor(raw_ftr).activate_motif_ratio_vec(to_add=nodes_and_edges),
                                      labels[name])
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
