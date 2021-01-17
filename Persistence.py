import gudhi
import matplotlib.pyplot as plt

class Persistence:
    def __init__(self, graph):
        simplex_tree = gudhi.SimplexTree()
        
        for node in graph:
            simplex_tree.insert([node], filtration = node)
            for el in (graph[node]):
                simplex_tree.insert([node, el], filtration = node + el)
        
        self.simplex_tree = simplex_tree
        
                
    def draw(self, betti = True):
        diag = self.simplex_tree.persistence(persistence_dim_max = True)
        
        if betti:
            print("Betti numbers [b_0, b_1, ...] = " + str(self.simplex_tree.betti_numbers()))
    
        
        ax = gudhi.plot_persistence_diagram(diag, legend=True)
        ax.set_title("Persistence diagram")
        ax.set_aspect("equal")
        plt.show()