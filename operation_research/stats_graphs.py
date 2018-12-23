import os
from collections import Counter
from bokeh.plotting import figure, save

from refael_learner import RefaelLearner
from parameters import REFAEL_PARAM


def statgraphs(learner: RefaelLearner):
    learner.data_loader().filter_by_nodes(min_nodes=REFAEL_PARAM['min_nodes']) if REFAEL_PARAM['min_nodes'] \
        else learner.data_loader().features_by_time
    db = learner.data_loader()
    db._load_database()
    # learner.stats_plot(db)
    multicolor_stats_plot(learner, db)


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
        save(figures[i], os.path.join(learner.base_dir(), "fig", "stats_graphs", titles[i] + ".html"))


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
        save(figures[i], os.path.join(learner.base_dir(), "fig", "stats_graphs_" + str(len(colors)), titles[i] +
                                      "_c2.html"))
