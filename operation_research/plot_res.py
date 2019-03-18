from bokeh.plotting import figure
from bokeh.io import export_png
import os
import pickle
from numpy.random import randint

COLORS = ['aqua', 'cadetblue', 'coral', 'darkgray', 'gold', 'lightseagreen', 'maroon'] * 2


def read_files(comp_criterion, dirname):
    res_dict = {}
    best_dict = {}
    rand_dict = {}
    perf_dict = {}
    for name in os.listdir(os.path.join("res", dirname, comp_criterion)):
        full_name = os.path.join("res", dirname, comp_criterion, name)
        s_name = name.split("_")
        if s_name[0] == "res":
            res_dict[(s_name[1], s_name[-1])] = pickle.load(open(full_name, "rb"))
        elif s_name[0] == "best":
            best_dict[(s_name[0], s_name[-1])] = pickle.load(open(full_name, "rb"))
        elif s_name[0] == "rand":
            rand_dict[(s_name[0], s_name[-1])] = pickle.load(open(full_name, "rb"))
        elif s_name[0] == "perf":
            if comp_criterion == "learn_method":
                perf_dict[(s_name[1], s_name[-1])] = pickle.load(open(full_name, "rb"))
            else:
                perf_dict[(s_name[0], s_name[-1])] = pickle.load(open(full_name, "rb"))
        else:
            continue
    return res_dict, best_dict, rand_dict, perf_dict


def plot_one_method(task, results, bests, rands, crit, figure_version, method="NN"):       # "XGBoost"
    if task == "recall":
        index = 1 
    elif task == "precision":
        index = 2
    else:
        index = 3
    criter = crit.split('_')[-1]
    if figure_version == "a1":
        new_data = [{key: results[key]} for key in results.keys() if key[0] == method]
        method = "XGBoost" if method == "XGBoost" else "NN"
        p = figure(plot_width=800, plot_height=350, title=task + " by time - " + crit + ": " + method,
                   x_axis_label="time", y_axis_label=task)
        if task == "recall":
            p.line(bests[('best', criter)][0], bests[('best', criter)][index], line_color="green", legend="best")
        p.line(rands[('rand', criter)][0], rands[('rand', criter)][index], line_color="red", legend="random")
        for k in range(len(new_data)):
            data = new_data[k]
            for i, (key, (x_axis, y_axis, z_axis, w_axis)) in enumerate(sorted(data.items(), key=lambda x: x[1])):
                crit_val = list(new_data[k].keys())[0][1]
                if task == "recall":
                    p.line(x_axis, y_axis, line_color=COLORS[k], legend=str(crit) + " = " + crit_val)
                elif task == "precision":
                    p.line(x_axis, z_axis, line_color=COLORS[k], legend=str(crit) + " = " + crit_val)
                else:
                    p.line(x_axis, w_axis, line_color=COLORS[k], legend=str(crit) + " = " + crit_val)
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
            p = figure(plot_width=800, plot_height=350, title=task + " by time - " + crit + " = " + crit_val +
                                                              ", method: " + method,
                       x_axis_label="time", y_axis_label=task)
            p.line(new_data[ind][(method, crit_val)][0], new_data[ind][(method, crit_val)][index], line_color="blue",
                   legend="model")
            if task == "recall":
                p.line(new_best[ind][('best', crit_val)][0], new_best[ind][('best', crit_val)][index],
                       line_color="green", legend="best")
            p.line(new_rand[ind][('rand', crit_val)][0], new_rand[ind][('rand', crit_val)][index], line_color="red",
                   legend="random")
            if not task == "false_alarm":
                p.legend.location = "bottom_right"
            else:
                p.legend.location = "top_left"
            p.toolbar.logo = None
            p.toolbar_location = None
            export_png(p, os.path.join("fig", crit, task + "_comparison_" + crit_val + ".png"))


def performance_plot(performance_dict, crit, figure_version, method="XGBoost"):       # "NN"
    for task in ['recall', 'precision', 'false_alarm']:
        if task == "recall":
            index = 1
        elif task == "precision":
            index = 2
        else:
            index = 3
        if figure_version == "a1":
            method = "XGBoost" if method == "XGBoost" else "NN"
            p = figure(plot_width=800, plot_height=350, title="All data preformance - " + task + " by time - " + crit +
                                                              ": " + method, x_axis_label="time", y_axis_label=task)
            shapes = [p.circle, p.square, p.triangle, p.diamond, p.cross, p.inverted_triangle]
            for i, (key, (x_axis, y_axis, z_axis, w_axis)) in enumerate(
                    sorted(performance_dict.items(), key=lambda x: x[1])):
                crit_val = key[1]
                if task == "recall":
                    shapes[i](x_axis, y_axis, line_color=COLORS[i],
                              legend=str(crit) + " = " + crit_val, size=10, alpha=0.5)
                elif task == "precision":
                    shapes[i](x_axis, z_axis, line_color=COLORS[i],
                              legend=str(crit) + " = " + crit_val, size=10, alpha=0.5)
                else:
                    shapes[i](x_axis, w_axis, line_color=COLORS[i],
                              legend=str(crit) + " = " + crit_val, size=10, alpha=0.5)
            p.legend.location = "bottom_center"
            p.legend.orientation = "horizontal"
            p.legend.label_text_font_size = "8pt"
            p.legend.margin = 3
            p.legend.padding = 4
            p.toolbar.logo = None
            p.toolbar_location = None
            export_png(p, os.path.join("fig", crit, task + "_performance_comparison.png"))
        else:
            new_perf = [{key: performance_dict[key]} for key in performance_dict.keys()]
            for ind in range(len(new_perf)):
                crit_val = list(new_perf[ind].keys())[0][1]
                p = figure(plot_width=800, plot_height=350,
                           title="All data performance - " + task + " by time - " + crit + " = " + crit_val +
                                 ", method: " + method, x_axis_label="time", y_axis_label=task)
                p.circle(new_perf[ind][('perf', crit_val)][0], new_perf[ind][('perf', crit_val)][index], color="blue",
                         size=12, alpha=0.5)

                p.toolbar.logo = None
                p.toolbar_location = None
                export_png(p, os.path.join("fig", crit, task + "_performance_" + crit_val + ".png"))


def compare_learning_methods(data, best, rand):
    p = figure(plot_width=800, plot_height=350, title="Recall by time - learn method comparison",
               x_axis_label="time", y_axis_label="recall")
    p.line(data[('XGBoost', 'method')][0], data[('XGBoost', 'method')][1], line_color="blue", legend="XGBoost")
    p.line(data[('NN', 'method')][0], data[('NN', 'method')][1], line_color="slategray", legend="NN")
    p.line(best[('best', 'method')][0], best[('best', 'method')][1], line_color="green", legend="best")
    p.line(rand[('rand', 'method')][0], rand[('rand', 'method')][1], line_color="red", legend="random")
    p.legend.location = "bottom_right"
    p.toolbar.logo = None
    p.toolbar_location = None
    export_png(p, os.path.join("fig", "learn_method", "recall_comparison.png"))

    q = figure(plot_width=800, plot_height=350, title="Precision by time - learn method comparison",
               x_axis_label="time", y_axis_label="precision")
    q.line(data[('XGBoost', 'method')][0], data[('XGBoost', 'method')][2], line_color="blue", legend="XGBoost")
    q.line(data[('NN', 'method')][0], data[('NN', 'method')][2], line_color="slategray", legend="NN")
    q.line(rand[('rand', 'method')][0], rand[('rand', 'method')][2], line_color="red", legend="random")
    q.legend.location = "bottom_right"
    q.toolbar.logo = None
    q.toolbar_location = None
    export_png(q, os.path.join("fig", "learn_method", "precision_comparison.png"))

    r = figure(plot_width=800, plot_height=350, title="False alarm by time - learn method comparison",
               x_axis_label="time", y_axis_label="false alarm")
    r.line(data[('XGBoost', 'method')][0], data[('XGBoost', 'method')][3], line_color="blue", legend="XGBoost")
    r.line(data[('NN', 'method')][0], data[('NN', 'method')][3], line_color="slategray", legend="NN")
    r.line(rand[('rand', 'method')][0], rand[('rand', 'method')][3], line_color="red", legend="random")
    r.legend.location = "top_left"
    r.toolbar.logo = None
    r.toolbar_location = None
    export_png(r, os.path.join("fig", "learn_method", "false_alarm_comparison.png"))


def learning_methods_performance(performance_dict):
    p = figure(plot_width=800, plot_height=350, title="Recall by time - learn method comparison on all data by time",
               x_axis_label="time", y_axis_label="recall")
    p.circle(performance_dict[('XGBoost', 'method')][0], performance_dict[('XGBoost', 'method')][1], color="blue",
             legend="XGBoost", size=10, alpha=0.5)
    p.triangle(performance_dict[('NN', 'method')][0], performance_dict[('NN', 'method')][1], color="peru",
               legend="NN", size=10, alpha=0.5)
    p.legend.location = "bottom_right"
    p.toolbar.logo = None
    p.toolbar_location = None
    export_png(p, os.path.join("fig", "learn_method", "recall_comparison_all_data.png"))

    q = figure(plot_width=800, plot_height=350, title="Precision by time - learn method comparison on all data by time",
               x_axis_label="time", y_axis_label="precision")
    q.circle(performance_dict[('XGBoost', 'method')][0], performance_dict[('XGBoost', 'method')][2], color="blue",
             legend="XGBoost", size=10, alpha=0.5)
    q.triangle(performance_dict[('NN', 'method')][0], performance_dict[('NN', 'method')][2], line_color="peru",
               legend="NN", size=10, alpha=0.5)
    q.legend.location = "bottom_right"
    q.toolbar.logo = None
    q.toolbar_location = None
    export_png(q, os.path.join("fig", "learn_method", "precision_comparison_all_data.png"))

    r = figure(plot_width=800, plot_height=350, title="False alarm by time - learn method comparison",
               x_axis_label="time", y_axis_label="false alarm")
    r.circle(performance_dict[('XGBoost', 'method')][0], performance_dict[('XGBoost', 'method')][3], line_color="blue",
             legend="XGBoost", size=10, alpha=0.5)
    r.triangle(performance_dict[('NN', 'method')][0], performance_dict[('NN', 'method')][3], line_color="peru",
               legend="NN", size=10, alpha=0.5)
    r.legend.location = "top_left"
    r.toolbar.logo = None
    r.toolbar_location = None
    export_png(r, os.path.join("fig", "learn_method", "false_alarm_comparison_all_data.png"))


if __name__ == "__main__":
    # Dictionary of whether to put all tries in one or to plot separated figures:
    method_format_dict = {"eps": "a1", "min_nodes": "s", "que_batch": "a1", "win": "s"}
    for criterion, fig_version in method_format_dict.items():
        res_data, best_data, rand_data, perf_data = read_files(criterion, "030319_184221")
        for tsk in ["recall", "precision", "false_alarm"]:
            plot_one_method(task=tsk, results=res_data, bests=best_data, rands=rand_data, crit=criterion,
                            figure_version=fig_version, method="XGBoost")
        performance_plot(performance_dict=perf_data, crit=criterion, figure_version=fig_version, method="XGBoost")
    res_, best_, rand_, perf_ = read_files("learn_method", "030319_184221")
    compare_learning_methods(res_, best_, rand_)
    learning_methods_performance(perf_)
