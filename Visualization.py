import networkx as nx
import matplotlib.pyplot as plt


class Visualization:
    def __init__(self, graph):
        G = nx.Graph()

        for node in graph:
            for edge in graph[node]:
                G.add_edge(node, edge)

        self.G = G

    def draw(self):
        options = {
            "font_size": 36,
            "node_size": 3000,
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
