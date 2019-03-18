import os
import pickle
import pandas as pd
import networkx as nx
import numpy as np
from collections import Counter
from bokeh.plotting import figure
from bokeh.io import export_png

from refael_learner import RefaelLearner
from parameters import REFAEL_PARAM


def statgraphs(learner: RefaelLearner):
    learner.data_loader().filter_by_nodes(min_nodes=REFAEL_PARAM['min_nodes']) if REFAEL_PARAM['min_nodes'] \
        else learner.data_loader().features_by_time
    db = learner.data_loader()
    db._load_database()
    stats_plot(learner, db)
    # multicolor_stats_plot(learner, db)


def stats_plot(learner: RefaelLearner, database):
    if not os.path.exists(os.path.join(learner.base_dir(), 'fig', 'stats_graphs')):
        os.mkdir(os.path.join(learner.base_dir(), 'fig', 'stats_graphs'))

    figures = []
    titles = ["Number of graphs over time", "Total number of nodes over time", "Total number of edges over time",
              "Average node degree over time", "Number of blacks over time", "Number of whites over time"]
    y_labels = ["number of graphs", "number of nodes", "number of edges", "average node degree", "number of blacks",
                "number of whites"]
    y_vals = []

    for i in range(6):
        figures.append(figure(plot_width=600, plot_height=250, title=titles[i], x_axis_label='time',
                              y_axis_label=y_labels[i]))
        y_vals.append([])

    for t in range(len(database.multi_graphs_by_time)):
        mg = database.multi_graphs_by_time[t]
        ids = mg._list_id
        valids = [gid for gid in ids if mg._graph_valid[gid]]

        y_vals[0].append(len(valids))
        y_vals[1].append(sum([mg.node_count(k) for k in valids]))
        y_vals[2].append(sum([mg.edge_count(k) for k in valids]))

        degs = []
        gnxs = [mg.get_gnx(gid) for gid in valids]
        for gnx in gnxs:
            degs = degs + [t[1] for t in list(gnx.degree([x for x in gnx.nodes()]))]
        avg_deg = sum(degs) / sum([mg.node_count(k) for k in valids])
        y_vals[3].append(avg_deg)

        labels = database._database._labels
        valid_labels = [labels[gr] for gr in valids]
        if REFAEL_PARAM['white_label']:
            y_vals[5].append(sum(valid_labels))
            y_vals[4].append(len(valid_labels) - sum(valid_labels))
        else:
            y_vals[4].append(sum(valid_labels))
            y_vals[5].append(len(valid_labels) - sum(valid_labels))
    for i in range(len(figures)):
        figures[i].line(list(range(len(database.multi_graphs_by_time))), y_vals[i], line_color='blue')
        export_png(figures[i], os.path.join(learner.base_dir(), "fig", "stats_graphs", titles[i] + "_mn_" +
                                            str(learner._params["min_nodes"]) + ".png"))


def multicolor_stats_plot(learner: RefaelLearner, database):
    labels = database._database._labels
    colors = list(Counter(labels.values()).keys())

    figures = []
    titles = ["Number of graphs over time", "Total number of nodes over time", "Total number of edges over time",
              "Average node degree over time"] + \
             ["Number of graphs of color " + str(color) + " over time" for color in colors]
    y_labels = ["number of graphs", "number of nodes", "number of edges", "average node degree"] + \
               ["number of color " + str(color) for color in colors]
    y_vals = []

    for i in range(4 + len(colors)):
        figures.append(figure(plot_width=600, plot_height=250, title=titles[i], x_axis_label='time',
                              y_axis_label=y_labels[i]))
        y_vals.append([])

    for t in range(len(database.multi_graphs_by_time)):
        mg = database.multi_graphs_by_time[t]
        ids = mg._list_id
        valids = [gid for gid in ids if mg._graph_valid[gid]]

        y_vals[0].append(len(valids))
        y_vals[1].append(sum([mg.node_count(k) for k in valids]))
        y_vals[2].append(sum([mg.edge_count(k) for k in valids]))

        degs = []
        gnxs = [mg.get_gnx(gid) for gid in valids]
        for gnx in gnxs:
            degs = degs + [t[1] for t in list(gnx.degree([x for x in gnx.nodes()]))]
        avg_deg = sum(degs) / sum([mg.node_count(k) for k in valids])
        y_vals[3].append(avg_deg)
        valid_labels = [labels[gr] for gr in valids]
        num_graphs_of_color = {c: 0 for c in colors}
        for i in range(len(valid_labels)):
            num_graphs_of_color[valid_labels[i]] += 1
        for val in range(len(num_graphs_of_color.keys())):
            y_vals[4 + val].append(num_graphs_of_color[colors[val]])
    if not os.path.exists(os.path.join(learner.base_dir(), 'fig', 'stats_graphs_' + str(len(colors)))):
        os.mkdir(os.path.join(learner.base_dir(), 'fig', 'stats_graphs_' + str(len(colors))))
    for i in range(len(figures)):
        figures[i].line(list(range(len(database.multi_graphs_by_time))), y_vals[i], line_color='blue')
        export_png(figures[i], os.path.join(learner.base_dir(), "fig", "stats_graphs_" + str(len(colors)),
                                            titles[i] + "_c2.png"))


def num_graphs_vs_min_nodes(learner_mn_list):
    # learner_mn_list: a list of tuples, (RefaelLearner, min_nodes)
    num_graphs_fig = figure(plot_width=800, plot_height=350,
                            title="Number of graphs (time) for several min. nodes values", x_axis_label='time',
                            y_axis_label="Number of graphs")
    num_blacks_fig = figure(plot_width=800, plot_height=350,
                            title="Number of blacks (time) for several min. nodes values", x_axis_label='time',
                            y_axis_label="Number of blacks")
    num_whites_fig = figure(plot_width=800, plot_height=350,
                            title="Number of whites (time) for several min. nodes values", x_axis_label='time',
                            y_axis_label="Number of whites")
    figs = [num_graphs_fig, num_blacks_fig, num_whites_fig]
    colors = ['blue', '', '', 'orange', '', 'red', '', '', 'purple', 'yellow', 'green']
    titles = ['graphs_per_time', 'blacks_per_time', 'whites_per_time']
    for learner, mn in learner_mn_list:
        if not os.path.exists(os.path.join(learner.base_dir(), 'fig', 'stats_graphs')):
            os.mkdir(os.path.join(learner.base_dir(), 'fig', 'stats_graphs'))
        learner.data_loader().filter_by_nodes(min_nodes=mn) if mn else learner.data_loader().features_by_time
        db = learner.data_loader()
        db._load_database()

        y_vals = [[], [], []]

        for t in range(len(db.multi_graphs_by_time)):
            mg = db.multi_graphs_by_time[t]
            ids = mg._list_id
            valids = [gid for gid in ids if mg._graph_valid[gid]]

            y_vals[0].append(len(valids))

            labels = db._database._labels
            valid_labels = [labels[gr] for gr in valids]
            if REFAEL_PARAM['white_label']:
                y_vals[2].append(sum(valid_labels))
                y_vals[1].append(len(valid_labels) - sum(valid_labels))
            else:
                y_vals[2].append(sum(valid_labels))
                y_vals[1].append(len(valid_labels) - sum(valid_labels))
        for i in range(len(figs)):
            figs[i].line(list(range(len(db.multi_graphs_by_time))), y_vals[i], line_color=colors[mn],
                         legend="min. nodes: " + str(mn))
            figs[i].legend.location = "top_left"
            figs[i].toolbar.logo = None
            figs[i].toolbar_location = None
    for i in range(len(figs)):
        export_png(figs[i], os.path.join(os.getcwd(), "..", "fig", "stats_graphs", str(titles[i]) + ".png"))


def node_deg_vs_min_nodes(learner_mn_list):
    # learner_mn_list: a list of tuples, (RefaelLearner, min_nodes)
    fig = figure(plot_width=800, plot_height=350, title="Average node degree (time) for several min. nodes values",
                 x_axis_label='time', y_axis_label="Average node degree")
    colors = ['blue', '', '', 'orange', '', 'red', '', '', 'purple', 'yellow', 'green']
    for learner, mn in learner_mn_list:
        if not os.path.exists(os.path.join(learner.base_dir(), 'fig', 'stats_graphs')):
            os.mkdir(os.path.join(learner.base_dir(), 'fig', 'stats_graphs'))
        learner.data_loader().filter_by_nodes(min_nodes=mn) if mn else learner.data_loader().features_by_time
        db = learner.data_loader()
        db._load_database()

        y_vals = []

        for t in range(len(db.multi_graphs_by_time)):
            mg = db.multi_graphs_by_time[t]
            ids = mg._list_id
            valids = [gid for gid in ids if mg._graph_valid[gid]]

            degs = []
            gnxs = [mg.get_gnx(gid) for gid in valids]
            for gnx in gnxs:
                degs = degs + [t[1] for t in list(gnx.degree([x for x in gnx.nodes()]))]
            avg_deg = sum(degs) / sum([mg.node_count(k) for k in valids])
            y_vals.append(avg_deg)

        fig.line(list(range(len(db.multi_graphs_by_time))), y_vals, line_color=colors[mn],
                 legend="min. nodes: " + str(mn))
        fig.legend.location = "top_left"
        fig.toolbar.logo = None
        fig.toolbar_location = None
    export_png(fig, os.path.join(os.getcwd(), "..", "fig", "stats_graphs", "avg_node_deg_per_time.png"))


def valid_binary_graphs():
    # Graphs with at least 3 vertices and at least one motif.
    # Have the needed gnxs in pkl.graph_measures
    dir_path = os.path.join(os.getcwd(),
                            "..", "pkl", "graph_measures", "Refael_12_18__bi_ds_1_st_0_ws_None_d_True_mc_False")
    fig_total = figure(plot_width=600, plot_height=250,
                       title="Number of graphs with at least 3 nodes and at least one motif",
                       x_axis_label='time', y_axis_label="number of graphs")
    fig_black = figure(plot_width=600, plot_height=250,
                       title="Number of blacks with at least 3 nodes and at least one motif",
                       x_axis_label='time', y_axis_label="number of graphs")
    fig_white = figure(plot_width=600, plot_height=250,
                       title="Number of whites with at least 3 nodes and at least one motif",
                       x_axis_label='time', y_axis_label="number of graphs")
    figs = [fig_white, fig_black, fig_total]
    titles = ['white_valids', 'black_valids', 'binary_valids']
    df = pd.read_csv(os.path.join(os.getcwd(), "..", "INPUT_DATABASE", "Refael_18_12_18_Binary.csv"))
    community_target = {}
    for line in range(df.shape[0]):
        if not df.loc[line, "Community"] in community_target:
            community_target[df.loc[line, "Community"]] = df.loc[line, "target"]
    colors = 2
    time_numbers = np.zeros((colors + 1, len(os.listdir(dir_path))))
    for time in os.listdir(dir_path):
        for community in os.listdir(os.path.join(dir_path, time)):
            graph = pickle.load(open(os.path.join(dir_path, time, community, "gnx.pkl"), "rb"))
            largest_cc = max(nx.weakly_connected_components(graph), key=len)
            if len(largest_cc) > 2:
                time_numbers[community_target[int(community)], int(time.split("_")[1])] += 1
                time_numbers[colors, int(time.split("_")[1])] += 1

    for i in range(len(figs)):
        figs[i].line(list(range(time_numbers.shape[1])), time_numbers[i, :])
        figs[i].toolbar.logo = None
        figs[i].toolbar_location = None
    for i in range(len(figs)):
        export_png(figs[i], os.path.join(os.getcwd(), "..", "fig", "stats_graphs", str(titles[i]) + ".png"))


def valid_multicolor_graphs():
    # Graphs with more than 3 vertices and at least on motif.
    # Have the needed gnxs in pkl.graph_measures
    # If color 2 is chosen, set labels to be 0 to whatever.
    dir_path = os.path.join(os.getcwd(), "..",
                            "pkl", "graph_measures", "Refael_12_18__c1_ds_1_st_0_ws_None_d_True_mc_False")
    community_target = {}
    df = pd.read_csv(os.path.join(os.getcwd(), "..", "INPUT_DATABASE", "Refael_18_12_18_Color_1.csv"))
    for line in range(df.shape[0]):
        if not df.loc[line, "Community"] in community_target:
            community_target[df.loc[line, "Community"]] = df.loc[line, "target"]

    colors = len(Counter(community_target.values()))
    figures = []
    titles = ["Number of graphs of color " + str(c) + " with at least 3 nodes and at least one motif"
              for c in range(colors)] + ["Total number of graphs with at least 3 nodes and at least one motif"]
    names = ["color_" + str(c) + "_valids" for c in range(colors)] + ["multicolor_valids"]

    for i in range(colors + 1):
        figures.append(figure(plot_width=600, plot_height=250, title=titles[i], x_axis_label='time',
                              y_axis_label="number_of_graphs"))

    time_numbers = np.zeros((colors + 1, len(os.listdir(dir_path))))
    for time in os.listdir(dir_path):
        for community in os.listdir(os.path.join(dir_path, time)):
            graph = pickle.load(open(os.path.join(dir_path, time, community, "gnx.pkl"), "rb"))
            largest_cc = max(nx.weakly_connected_components(graph), key=len)
            if len(largest_cc) > 2:
                time_numbers[int(community_target[int(community)]), int(time.split("_")[1])] += 1
                time_numbers[colors, int(time.split("_")[1])] += 1

    for i in range(len(figures)):
        figures[i].line(list(range(time_numbers.shape[1])), time_numbers[i, :])
        figures[i].toolbar.logo = None
        figures[i].toolbar_location = None
    for i in range(len(figures)):
        export_png(figures[i], os.path.join(os.getcwd(), "..", "fig", "stats_graphs", str(names[i]) + ".png"))


if __name__ == "__main__":
    learner_mn = []
    # for mnn in [0, 3, 5, 10]:
    #     r = RefaelLearner()
    #     learner_mn.append((r, mnn))
    # num_graphs_vs_min_nodes(learner_mn)
    # node_deg_vs_min_nodes(learner_mn)
    valid_binary_graphs()
    # valid_multicolor_graphs()
    # r = RefaelLearner()
    # statgraphs(r)
