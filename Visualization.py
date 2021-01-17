import networkx as nx
import matplotlib.pyplot as plt


class Visualization:
    def __init__(self, graph):
        G = nx.Graph()

        for node in graph:
            G.add_node(node)
            for edge in graph[node]:
                G.add_edge(node, edge)

        self.G = G

    def draw(self, font_size, node_size):
        options = {
            "font_size": font_size,
            "node_size": node_size,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 5,
            "width": 5,
        }
        nx.draw_networkx(self.G, **options)

        ax = plt.gca()
        ax.margins(0.20)
        plt.axis("off")
        plt.show()
