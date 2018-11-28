import os
from active_learning import TimedActiveLearning
from data_loader import DataLoader
from parameters import REFAEL_PARAM, ACTIVE_LEARNING, MACHINE_LEARNING
from refael_ml import RefaelML


class RefaelLearner:
    def __init__(self):
        self._params = REFAEL_PARAM
        self._base_dir = __file__.replace("/", os.sep)
        self._base_dir = os.path.join(self._base_dir.rsplit(os.sep, 1)[0])
        self._data_loader = DataLoader(self._params)

    def run_ml(self):
        data = self._data_loader.filter_by_nodes(min_nodes=self._params['min_nodes']) if self._params['min_nodes'] \
            else self._data_loader.features_by_time
        RefaelML(self._params, data).run()

    def run_al(self):
        data = self._data_loader.filter_by_nodes(min_nodes=self._params['min_nodes']) if self._params['min_nodes'] \
            else self._data_loader.features_by_time
        al = TimedActiveLearning(data, self._params)
        al.run()
        # al.pred_vs_label()
        # al.plot()
        al.tziurim()

if __name__ == "__main__":
    r = RefaelLearner()
    if ACTIVE_LEARNING:
        r.run_al()

    if MACHINE_LEARNING:
        r.run_ml()
    end = 0

