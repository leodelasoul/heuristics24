import matplotlib.pyplot as plt
import networkx as nx


text = ""
def draw_instance(u, x, w):
    graph = nx.Graph()

    graph.add_nodes_from(u, bipartite=0)  # Top nodes
    graph.add_nodes_from(x, bipartite=1)  # Bottom nodes

    top_nodes = [i for i in range(1, u.size + 1)]
    w: list[list[int]] = w
    draw_big_instance(w, graph, top_nodes) if len(w) > 50 else draw_small_instance(w, graph, top_nodes)


def draw_big_instance(w, graph, top_nodes):
    edges = []
    for u, i in enumerate(w[1:]):
        for v_index, weight in enumerate(i):
            if weight > 0:
                edges.append((u + 1, v_index))
    graph.add_edges_from(edges)
    pos = nx.bipartite_layout(graph, top_nodes)
    rotated_pos = {node: (y, -x) for node, (x, y) in pos.items()}
    ax = plt.gca()
    ax.set_title(text)
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
    ax = plt.gca()
    ax.set_title(text)
    nx.draw(graph, rotated_pos, with_labels=True,
            node_color=["skyblue" if node in top_nodes else "lightgreen" for node in graph.nodes()], ax=ax)
    nx.draw_networkx_edge_labels(graph, rotated_pos, edge_labels=weights)  # Display weights
    plt.show()


def convergence_plot(data):
    iterations = data['iteration']  # Iterations up to the best solution
    objective_vals = data['obj_val']  # Iterations up to the best solution

    # Plotting the convergence plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, objective_vals, label="Best Objective Value", color="blue", linewidth=2)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Objective Value", fontsize=12)
    plt.title("Convergence Plot", fontsize=14)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()