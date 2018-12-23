import datetime

from parameters import REFAEL_PARAM
from refael_learner import RefaelLearner
import pickle
import os

LEARN_METHOD = "learn_method"
EPS = "eps"
MIN_NODES = "min_nodes"
QUE_BATCH = "que_batch"
WIN = "win"


def grid_learn_method(base_name):
    base = os.path.join("res", base_name, LEARN_METHOD)
    print("XG_boost")
    REFAEL_PARAM['learn_method'] = "XG_Boost"
    res, best = RefaelLearner().run_al()
    pickle._dump(best, open(os.path.join(base, "best_learn_method"), "wb"))
    pickle._dump(res, open(os.path.join(base, "res_XG_Boost_learn_method"), "wb"))
    print("NN")
    REFAEL_PARAM['learn_method'] = "nn"
    res, best = RefaelLearner().run_al()
    pickle._dump(res, open(os.path.join(base, "res_NN_learn_method"), "wb"))
    pickle._dump(best, open(os.path.join(base, "best_NN_learn_method"), "wb"))
    REFAEL_PARAM['learn_method'] = "XG_Boost"


def grid_slide_window(base_name, win_size_list: list):
    base = os.path.join("res", base_name, WIN)
    print("XG_boost")
    default_start = REFAEL_PARAM['start_time']
    default_win_size = REFAEL_PARAM['window_size']

    for i in win_size_list:
        REFAEL_PARAM['start_time'] = i
        REFAEL_PARAM['window_size'] = i
        res, best = RefaelLearner().run_al()
        pickle._dump(best, open(os.path.join(base, "best_win_" + str(i)), "wb"))
        pickle._dump(res, open(os.path.join(base, "res_XG_Boost_win_" + str(i)), "wb"))

    REFAEL_PARAM['start_time'] = default_start
    REFAEL_PARAM['window_size'] = None


def grid_eps(base_name, eps_list: list):
    base = os.path.join("res", base_name, EPS)

    default_eps = REFAEL_PARAM['eps']

    for i in eps_list:
        REFAEL_PARAM['eps'] = i
        res, best = RefaelLearner().run_al()
        pickle._dump(best, open(os.path.join(base, "best_eps_" + str(i)), "wb"))
        pickle._dump(res, open(os.path.join(base, "res_XG_Boost_eps_" + str(i)), "wb"))

    REFAEL_PARAM['eps'] = default_eps


def grid_bach_size(base_name, query_batch_list : list):
    base = os.path.join("res", base_name, QUE_BATCH)
    default_que = REFAEL_PARAM['queries_per_time']
    default_batch = REFAEL_PARAM['batch_size']

    for queries_per_time, batch_size in query_batch_list:
        REFAEL_PARAM['queries_per_time'] = queries_per_time
        REFAEL_PARAM['batch_size'] = batch_size
        res, best = RefaelLearner().run_al()
        pickle._dump(best,
                     open(os.path.join(base, "best_queries_" + str(queries_per_time) + "_batch_" + str(batch_size)), "wb"))
        pickle._dump(res, open(
            os.path.join(base, "res_XG_Boost_queries_" + str(queries_per_time) + "_batch_" + str(batch_size)), "wb"))

    REFAEL_PARAM['queries_per_time'] = default_que
    REFAEL_PARAM['batch_size'] = default_batch


def grid_min_nodes(base_name, min_nodes_list: list):
    base = os.path.join("res", base_name, MIN_NODES)
    default_min_nodes = REFAEL_PARAM['min_nodes']

    for num_nodes in min_nodes_list:
        REFAEL_PARAM['min_nodes'] = num_nodes
        res, best = RefaelLearner().run_al()
        pickle._dump(best, open(os.path.join(base, "best_min_nodes_" + str(num_nodes)), "wb"))
        pickle._dump(res, open(os.path.join(base, "res_XG_Boost_min_nodes_" + str(num_nodes)), "wb"))

    REFAEL_PARAM['min_nodes'] = default_min_nodes


def create_res_dir(base_name):
    os.mkdir(os.path.join("res", base_name))
    os.mkdir(os.path.join("res", base_name, EPS))
    os.mkdir(os.path.join("res", base_name, LEARN_METHOD))
    os.mkdir(os.path.join("res", base_name, MIN_NODES))
    os.mkdir(os.path.join("res", base_name, QUE_BATCH))
    os.mkdir(os.path.join("res", base_name, WIN))


if __name__ == "__main__":
    base_name = str(datetime.datetime.now())
    create_res_dir(base_name)

    # grid_slide_window(base_name, [1, 5, 15])
    # grid_bach_size(base_name, [(1, 8), (2, 4), (8, 1)])
    # grid_eps(base_name, [0.01, 0.1,  0.3])
    # grid_min_nodes(base_name, [5, 10, 15])
    # grid_learn_method(base_name)
    #
    # grid_slide_window(base_name, [1, 5, 15])
    # grid_bach_size(base_name, [(1, 4), (2, 2), (4, 1)])
    # grid_eps(base_name, [0.01, 0.1, 0.3])
    # grid_min_nodes(base_name, [5, 10, 15])
    grid_learn_method(base_name)
