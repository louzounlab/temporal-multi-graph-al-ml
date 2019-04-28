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
    print("XGBoost")
    REFAEL_PARAM['learn_method'] = "XG_Boost"
    # res, best, rand, perf = RefaelLearner().run_al(rand=True)
    res, best, rand, perf = RefaelLearner().run_al_bi_avg(rand=True)
    pickle._dump(best, open(os.path.join(base, "best_learn_method"), "wb"))
    pickle._dump(res, open(os.path.join(base, "res_XGBoost_learn_method"), "wb"))
    pickle._dump(rand, open(os.path.join(base, "rand_learn_method"), "wb"))
    pickle._dump(perf, open(os.path.join(base, "perf_XGBoost_learn_method"), "wb"))
    print("NN")
    REFAEL_PARAM['learn_method'] = "nn"
    # res, _, _, perf = RefaelLearner().run_al()
    res, _, _, perf = RefaelLearner().run_al_bi_avg()
    pickle._dump(res, open(os.path.join(base, "res_NN_learn_method"), "wb"))
    pickle._dump(perf, open(os.path.join(base, "perf_NN_learn_method"), "wb"))
    REFAEL_PARAM['learn_method'] = "XG_Boost"


def grid_slide_window(base_name, win_size_list: list):
    base = os.path.join("res", base_name, WIN)
    print("XGBoost")
    default_start = REFAEL_PARAM['start_time']
    default_win_size = REFAEL_PARAM['window_size']

    for i in win_size_list:
        # REFAEL_PARAM['start_time'] = i
        REFAEL_PARAM['window_size'] = i
        res, best, rand, perf = RefaelLearner().run_al_bi_avg(rand=True)
        pickle._dump(best, open(os.path.join(base, "best_win_" + str(i)), "wb"))
        pickle._dump(res, open(os.path.join(base, "res_XGBoost_win_" + str(i)), "wb"))
        pickle._dump(rand, open(os.path.join(base, "rand_win_" + str(i)), "wb"))
        pickle._dump(perf, open(os.path.join(base, "perf_win_" + str(i)), "wb"))

    REFAEL_PARAM['start_time'] = default_start
    REFAEL_PARAM['window_size'] = default_win_size


def grid_eps(base_name, eps_list: list):
    base = os.path.join("res", base_name, EPS)

    default_eps = REFAEL_PARAM['eps']

    for i in eps_list:
        REFAEL_PARAM['eps'] = i
        if i == eps_list[0]:
            res, best, rand, perf = RefaelLearner().run_al_bi_avg(rand=True)
            pickle._dump(best, open(os.path.join(base, "best_eps"), "wb"))
            pickle._dump(res, open(os.path.join(base, "res_XGBoost_eps_" + str(i)), "wb"))
            pickle._dump(rand, open(os.path.join(base, "rand_eps"), "wb"))
            pickle._dump(perf, open(os.path.join(base, "perf_XGBoost_eps_" + str(i)), "wb"))
        else:
            res, best, _, perf = RefaelLearner().run_al_bi_avg()
            pickle._dump(res, open(os.path.join(base, "res_XGBoost_eps_" + str(i)), "wb"))
            pickle._dump(perf, open(os.path.join(base, "perf_XGBoost_eps_" + str(i)), "wb"))
    REFAEL_PARAM['eps'] = default_eps


def grid_batch_size(base_name, query_batch_list: list):
    base = os.path.join("res", base_name, QUE_BATCH)
    default_que = REFAEL_PARAM['queries_per_time']
    default_batch = REFAEL_PARAM['batch_size']

    for queries_per_time, batch_size in query_batch_list:
        REFAEL_PARAM['queries_per_time'] = queries_per_time
        REFAEL_PARAM['batch_size'] = batch_size
        if (queries_per_time, batch_size) == query_batch_list[0]:
            res, best, rand, perf = RefaelLearner().run_al_bi_avg(rand=True)
            pickle._dump(best, open(
                os.path.join(base, "best_query_batch"), "wb"))
            pickle._dump(res, open(os.path.join(
                base, "res_XGBoost_query" + str(queries_per_time) + "batch" + str(batch_size)), "wb"))
            pickle._dump(rand, open(
                os.path.join(base, "rand_query_batch"), "wb"))
            pickle._dump(perf, open(os.path.join(
                base, "perf_XGBoost_query" + str(queries_per_time) + "batch" + str(batch_size)), "wb"))
        else:
            res, _, _, perf = RefaelLearner().run_al_bi_avg()
            pickle._dump(res, open(os.path.join(
                base, "res_XGBoost_query" + str(queries_per_time) + "batch" + str(batch_size)), "wb"))
            pickle._dump(perf, open(os.path.join(
                base, "perf_XGBoost_query" + str(queries_per_time) + "batch" + str(batch_size)), "wb"))
    REFAEL_PARAM['queries_per_time'] = default_que
    REFAEL_PARAM['batch_size'] = default_batch


def grid_min_nodes(base_name, min_nodes_list: list):
    min_nodes_to_batch_needed = {0: 20, 3: 15, 5: 8, 10: 4}
    base = os.path.join("res", base_name, MIN_NODES)
    default_min_nodes = REFAEL_PARAM['min_nodes']

    for num_nodes in min_nodes_list:
        REFAEL_PARAM['min_nodes'] = num_nodes
        REFAEL_PARAM['batch_size'] = min_nodes_to_batch_needed[num_nodes]  # fewer graphs ==> fewer guesses
        res, best, rand, perf = RefaelLearner().run_al_bi_avg(rand=True)
        pickle._dump(best, open(os.path.join(base, "best_min_nodes_" + str(num_nodes)), "wb"))
        pickle._dump(res, open(os.path.join(base, "res_XGBoost_min_nodes_" + str(num_nodes)), "wb"))
        pickle._dump(rand, open(os.path.join(base, "rand_min_nodes_" + str(num_nodes)), "wb"))
        pickle._dump(perf, open(os.path.join(base, "perf_XGBoost_min_nodes_" + str(num_nodes)), "wb"))

    REFAEL_PARAM['min_nodes'] = default_min_nodes


def create_res_dir(base_name):
    os.makedirs(os.path.join("res", base_name))
    os.mkdir(os.path.join("res", base_name, EPS))
    os.mkdir(os.path.join("res", base_name, LEARN_METHOD))
    os.mkdir(os.path.join("res", base_name, MIN_NODES))
    os.mkdir(os.path.join("res", base_name, QUE_BATCH))
    os.mkdir(os.path.join("res", base_name, WIN))


if __name__ == "__main__":
    # base_name_ = str(datetime.datetime.now().strftime('%d%m%y_%H%M%S'))
    base_name_ = "030319_184221"
    # create_res_dir(base_name_)

    grid_slide_window(base_name_, [1, 5, 10])
    # grid_batch_size(base_name_, [(1, 15), (3, 5), (5, 3), (15, 1)])
    # grid_eps(base_name_, [0, 0.01, 0.1,  0.3, 1])
    # grid_min_nodes(base_name_, [0, 3, 5, 10])
    # grid_learn_method(base_name_)

    # grid_slide_window(base_name, [1, 5, 15])
    # grid_batch_size(base_name, [(1, 4), (2, 2), (4, 1)])
    # grid_eps(base_name, [0.01, 0.1, 0.3])
    # grid_min_nodes(base_name, [5, 10, 15])
    # grid_learn_method(base_name)
