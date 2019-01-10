from bokeh.plotting import figure
from bokeh.io import export_png
import os
import pickle
from numpy.random import randint

COLORS = ['aqua', 'cadetblue', 'coral', 'darkgray', 'gold', 'lightseagreen', 'maroon']


def read_files(comp_criterion):
    res_dict = {}
    best_dict = {}
    rand_dict = {}
    for name in os.listdir(os.path.join("res", "241218_161454", comp_criterion)):
        full_name = os.path.join("res", "241218_161454", comp_criterion, name)
        s_name = name.split("_")
        if s_name[0] == "res":
            res_dict[(s_name[1], s_name[-1])] = pickle.load(open(full_name, "rb"))
        elif s_name[0] == "best":
            best_dict[(s_name[0], s_name[-1])] = pickle.load(open(full_name, "rb"))
        elif s_name[0] == "rand":
            rand_dict[(s_name[0], s_name[-1])] = pickle.load(open(full_name, "rb"))
        else:
            continue
    return res_dict, best_dict, rand_dict


def plot_one_method(task, results, bests, rands, crit, figure_version, method="NN"):       # "XGBoost"
    index = 1 if task == "recall" else 2
    criter = crit.split('_')[-1]
    if figure_version == "a1":
        new_data = [{key: results[key]} for key in results.keys() if key[0] == method]
        method = "XGBoost" if method == "XGBoost" else "NN"
        p = figure(plot_width=600, plot_height=250, title=task + " by time - " + crit + ": " + method,
                   x_axis_label="revealed:  (time*batch_size)/total_communities", y_axis_label="recall")
        if task == "recall":
            p.line(bests[('best', criter)][0], bests[('best', criter)][index], line_color="green", legend="best")
        p.line(rands[('rand', criter)][0], rands[('rand', criter)][index], line_color="red", legend="random")
        for k in range(len(new_data)):
            data = new_data[k]
            for i, (key, (x_axis, y_axis, z_axis)) in enumerate(sorted(data.items(), key=lambda x: x[1])):
                if task == "recall":
                    p.line(x_axis, y_axis, line_color=COLORS[k], legend=str(key[1]))
                else:
                    p.line(x_axis, z_axis, line_color=COLORS[k], legend=str(key[1]))
        p.legend.location = "bottom_right"
        if crit == "que_batch":
            p.legend.label_text_font_size = "8pt"
            p.legend.margin = 3
            p.legend.padding = 4
        p.legend.click_policy = "hide"
        p.toolbar.logo = None
        p.toolbar_location = None
        export_png(p, os.path.join("fig", crit, task + "_comparison.png"))
    else:
        new_data = [{key: results[key]} for key in results.keys() if key[0] == method]
        new_best = [{key: bests[key]} for key in bests.keys()]
        new_rand = [{key: rands[key]} for key in rands.keys()]
        for ind in range(len(new_data)):
            crit_val = list(new_data[ind].keys())[0][1]
            method = "XGBoost" if method == "XGBoost" else "NN"
            p = figure(plot_width=600, plot_height=250, title=task + " by time - " + crit + " = " + crit_val +
                                                              ", method: " + method,
                       x_axis_label="revealed:  (time*batch_size)/total_communities", y_axis_label="recall")
            p.line(new_data[ind][(method, crit_val)][0], new_data[ind][(method, crit_val)][index], line_color="blue",
                   legend="recall")
            if task == "recall":
                p.line(new_best[ind][('best', crit_val)][0], new_best[ind][('best', crit_val)][index],
                       line_color="green", legend="best")
            p.line(new_rand[ind][('rand', crit_val)][0], new_rand[ind][('rand', crit_val)][index], line_color="red",
                   legend="random")
            p.legend.location = "bottom_right"
            p.legend.click_policy = "hide"
            p.toolbar.logo = None
            p.toolbar_location = None
            export_png(p, os.path.join("fig", crit, task + "_comparison_" + crit_val + ".png"))


def compare_learning_methods(data, best, rand):
    p = figure(plot_width=600, plot_height=250, title="Recall by time - learn method comparison",
               x_axis_label="revealed:  (time*batch_size)/total_communities", y_axis_label="recall")
    p.line(data[('XGBoost', 'method')][0], data[('XGBoost', 'method')][1], line_color="blue", legend="XGBoost")
    p.line(data[('NN', 'method')][0], data[('NN', 'method')][1], line_color="slategray", legend="NN")
    p.line(best[('best', 'method')][0], best[('best', 'method')][1], line_color="green", legend="best")
    p.line(rand[('rand', 'method')][0], rand[('rand', 'method')][1], line_color="red", legend="random")
    p.legend.location = "bottom_right"
    p.legend.click_policy = "hide"
    p.toolbar.logo = None
    p.toolbar_location = None
    export_png(p, os.path.join("fig", "learn_method", "recall_comparison.png"))

    q = figure(plot_width=600, plot_height=250, title="Precision by time - learn method comparison",
               x_axis_label="revealed:  (time*batch_size)/total_communities", y_axis_label="precision")
    q.line(data[('XGBoost', 'method')][0], data[('XGBoost', 'method')][2], line_color="blue", legend="XGBoost")
    q.line(data[('NN', 'method')][0], data[('NN', 'method')][2], line_color="slategray", legend="NN")
    q.line(rand[('rand', 'method')][0], rand[('rand', 'method')][2], line_color="red", legend="random")
    q.legend.location = "bottom_right"
    q.legend.click_policy = "hide"
    q.toolbar.logo = None
    q.toolbar_location = None
    export_png(q, os.path.join("fig", "learn_method", "precision_comparison.png"))


if __name__ == "__main__":
    # Dictionary of whether to put all tries in one or to plot separated figures:
    method_format_dict = {"eps": "a1", "min_nodes": "s", "que_batch": "a1", "win": "s"}
    for criterion, fig_version in method_format_dict.items():
        res_data, best_data, rand_data = read_files(criterion)
        # plot_one_method(task="recall", results=res_data, bests=best_data, rands=rand_data, crit=criterion,
        #                 figure_version=fig_version, method="XGBoost")
        plot_one_method(task="precision", results=res_data, bests=best_data, rands=rand_data, crit=criterion,
                        figure_version=fig_version, method="XGBoost")
    res_, best_, rand_ = read_files("learn_method")
    compare_learning_methods(res_, best_, rand_)

