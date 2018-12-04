import os
from active_learning import TimedActiveLearning
from data_loader import DataLoader
from parameters import REFAEL_PARAM, ACTIVE_LEARNING, MACHINE_LEARNING
from refael_ml import RefaelML


class RefaelLearner:
    def __init__(self):
        self._params = REFAEL_PARAM
        self._params['database_full_name'] = self._full_name()
        self._base_dir = __file__.replace("/", os.sep)
        self._base_dir = os.path.join(self._base_dir.rsplit(os.sep, 1)[0])
        self._data_loader = DataLoader(self._params)

    def _full_name(self):
        return self._params['database'] + "__ds_" + str(self._params['days_split']) + "_st_" + \
               str(self._params['start_time']) + "_ws_" + str(self._params['window_size']) + "_d_" + \
               str(self._params['directed']) + "_mc_" + str(self._params['max_connected'])

    def run_ml(self):
        data = self._data_loader.filter_by_nodes(min_nodes=self._params['min_nodes']) if self._params['min_nodes'] \
            else self._data_loader.features_by_time
        RefaelML(self._params, data).run()

    def run_al(self):
        data = self._data_loader.filter_by_nodes(min_nodes=self._params['min_nodes']) if self._params['min_nodes'] \
            else self._data_loader.features_by_time
        al = TimedActiveLearning(data, self._params)
        best = al.best_recall_plot()
        res = al.run()
        al.plot()
        return res, best


if __name__ == "__main__":

    r = RefaelLearner()
    if ACTIVE_LEARNING:
        r.run_al()

    if MACHINE_LEARNING:
        r.run_ml()
    end = 0

