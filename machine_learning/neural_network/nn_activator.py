import torch
import copy
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch import nn
from machine_learning.neural_network.nn_dataset import DynamicDataManager


# class to train models
class FeedForwardNet:
    def __init__(self, model: nn.Module, train_size=0.8, gpu=False):
        self._len_data = 0
        self._train_size = train_size
        self._gpu = gpu
        # init models with current models
        self._model = model
        if self._gpu:
            self._model.cuda()
        # empty list for every model - y axis for plotting loss by epochs
        self._test_loader = None
        self._train_loader = None
        self._train_validation = None
        self._data_manager = None

    # load dataset
    def update_data(self, components, labels):
        # randomly split the train into test and validation
        if not self._data_manager:
            self._data_manager = DynamicDataManager(components, labels, train_size=self._train_size, gpu=self._gpu)
        else:
            self._data_manager.update(components, labels)

        # set train loader
        self._train_loader = DataLoader(
            self._data_manager.train,
            batch_size=64, shuffle=True
        )

        # set train loader
        self._train_validation = DataLoader(
            self._data_manager.train,
            batch_size=1, shuffle=False
        )

        # set validation loader
        self._test_loader = DataLoader(
            self._data_manager.test,
            batch_size=1, shuffle=False
        )

    # train a model, input is the enum of the model type
    def train(self, total_epoch, early_stop=None, validation_rate=1, reset_optimizer_lr=False, stop_auc=False):
        if reset_optimizer_lr:
            self._model.set_optimizer(lr=reset_optimizer_lr)
        # EarlyStopping
        best_model = copy.deepcopy(self._model)
        best_auc = 0
        prev_auc = 0
        curr_auc = 0
        count_no_improvement = 0
        auc_res = []        # return auc

        for epoch_num in range(total_epoch):
            # set model to train mode
            self._model.train()

            # calc number of iteration in current epoch
            for batch_index, (data, label) in enumerate(self._train_loader):
                # print progress
                self._model.zero_grad()                         # zero gradients
                output = self._model(data)                      # calc output of current model on the current batch
                loss = F.binary_cross_entropy(output, label)    # define loss node ( negative log likelihood)
                loss.backward()                                 # back propagation
                self._model.optimizer.step()                    # update weights

            t = "Train"

            if stop_auc:
                curr_auc_test, curr_loss = self._validate(self._test_loader)
                if best_auc < curr_auc_test < 1:
                    best_auc = curr_auc_test
                    best_model = copy.deepcopy(self._model)

            # validate
            if validation_rate and epoch_num % validation_rate == 0:
                curr_auc, curr_loss = self._validate(self._train_validation)
                auc_res.append(curr_auc)

                print(str(epoch_num) + "/" + str(total_epoch))
                print(t + " --- validation results = auc:\t" + str(curr_auc) + "\tavg_loss:\t" + str(curr_loss)) if \
                    curr_auc >= 0 else print(t + " --- AUC_Error")

            # EarlyStopping
            count_no_improvement = 0 if prev_auc < curr_auc else count_no_improvement + 1
            prev_auc = curr_auc
            if early_stop and count_no_improvement > early_stop:
                break

        if stop_auc and best_auc > 0:
            self._model = best_model
        return auc_res

    # validation function only the model and the data are important for input, the others are just for print
    def _validate(self, data_loader):
        loss_count = 0
        self._model.eval()

        y_true_auc = []
        y_score_auc = []

        # run though all examples in validation set (from input)
        for batch_index, (data, label) in enumerate(data_loader):

            output = self._model(data)                                          # calc output of the model
            loss_count += F.binary_cross_entropy(output, label).item()          # sum total loss of all iteration

            y_true_auc.append(label.data.tolist()[0])
            y_score_auc.append(output.data.tolist()[0][0])

        # add loss to y axis + print accuracy and average loss
        # acc = float(correct_count) / len(data_loader.dataset)
        loss = float(loss_count / len(data_loader.dataset))
        try:
            auc = roc_auc_score(y_true_auc, y_score_auc)
        except ValueError:
            auc = -1
        return auc, loss

    # Test
    def test(self):
        auc, loss =  self._validate(self._test_loader)
        print("Test --- validation results = auc:\t" + str(auc) + "\tavg_loss:\t" + str(loss)) if \
            auc >= 0 else print("Test --- AUC_Error")

    def predict(self, data_dict):
        results = {}
        for name, vec in data_dict.items():
            vec = torch.Tensor(vec)
            if self._gpu:
                vec.cuda()
            output = self._model(vec)                                          # calc output of the model
            results[name] = output
        return results


# (self, num_layers=3, layers_dim=(223, 140, 70), train_size=0.8, lr=0.0001, batch_norm=True, drop_out=0.3
#                  , l2_penalty=0.005, activation_func="relu", gpu=False):