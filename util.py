import matplotlib.pyplot as plt
import networkx as nx
import numpy
import numpy as np
def draw_big_instance(w, graph, top_nodes):
    edges = []
    for u, i in enumerate(w[1:]):
        for v_index, weight in enumerate(i):
            if weight > 0:
                edges.append((u + 1, v_index))
    graph.add_edges_from(edges)
    pos = nx.bipartite_layout(graph, top_nodes)
    rotated_pos = {node: (y, -x) for node, (x, y) in pos.items()}
    nx.draw(graph, rotated_pos, with_labels=True,
            node_color=["skyblue" if node in top_nodes else "lightgreen" for node in graph.nodes()])
    plt.show()


def draw_small_instance(w, graph, top_nodes):
    edges = []
    for u, i in enumerate(w[1:]):
        for v_index, weight in enumerate(i):
            if weight > 0:
                edges.append((u + 1, v_index, weight))

    graph.add_weighted_edges_from(edges)
    pos = nx.bipartite_layout(graph, top_nodes)
    rotated_pos = {node: (y, -x) for node, (x, y) in pos.items()}
    weights = nx.get_edge_attributes(graph, 'weight')  # Extract weights

    nx.draw(graph, rotated_pos, with_labels=True,
            node_color=["skyblue" if node in top_nodes else "lightgreen" for node in graph.nodes()])
    nx.draw_networkx_edge_labels(graph, rotated_pos, edge_labels=weights)  # Display weights
    plt.show()