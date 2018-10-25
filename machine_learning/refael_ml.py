from enum import Enum
import numpy as np
from machine_learning.neural_network.nn_activator import FeedForwardNet
from nn_models import NeuralNet3

FTR_IDX = 0
LABEL_IDX = 1


class LearningMethod(Enum):
    RF = "random_forest"
    SVM = "support_vector_machine"
    FEED_FORWARD_NN = "feed_forward_neural_network"
    XG_BOOST = 'XG-Boost'


class RefaelML:
    def __init__(self, params, time_list):
        self._time_list = time_list
        self._len = len(time_list)
        self._method = params['learn_method'] if 'learn_method' in params else LearningMethod.FEED_FORWARD_NN
        self._params = params

    def run(self):
        if self._method.value == LearningMethod.FEED_FORWARD_NN.value:
            self._neural_net()

    def _matrix_for_time(self, time_idx: int, key_func=None):
        mx = np.vstack([self._time_list[time_idx][key][FTR_IDX] for key in sorted(self._time_list[time_idx], key=key_func)])
        labels = [self._time_list[time_idx][key][LABEL_IDX] for key in sorted(self._time_list[time_idx], key=key_func)]
        return mx, labels

    def _dict_for_time(self, time_idx: int):
        ftr = {}
        labels = {}
        for name, ftr_tupple in self._time_list[time_idx].items():
            ftr[name] = ftr_tupple[FTR_IDX]
            labels[name] = ftr_tupple[LABEL_IDX]
        return ftr, labels

    def _neural_net(self):
        early_stop = self._params['early_stop'] if 'early_stop' in self._params else 200
        epochs = self._params['epochs'] if 'epochs' in self._params else 300

        model = NeuralNet3(layers_size=(225, 150, 75), lr=0.001)
        network = FeedForwardNet(model, train_size=0.7, gpu=False)

        for i in range(self._len):
            print("-----------------------------------    TIME " + str(i) + "    -------------------------------------")
            components, labels = self._dict_for_time(i)
            network.update_data(components, labels)
            network.train(epochs, early_stop=early_stop, validation_rate=200)
            network.test()



