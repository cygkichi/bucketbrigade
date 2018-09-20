import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph

sample_state = np.array([[1,0], [0,1], [0,0], [0,0], [0,0]
                         , [0,0], [0,0], [0,0], [0,0]], dtype = int)
sample_cand_edges = np.array(
[
    [[0,1], [2,3], [1,4], [3,3], [4,5], [5,6], [7,8], [6,0], [8,2]],
    [[0,1], [2,3], [1,4], [3,3], [4,7], [5,6], [7,8], [6,0], [8,2]],
    [[0,1], [2,3], [1,1], [3,4], [4,5], [5,6], [7,8], [6,0], [8,2]],
    [[0,1], [2,3], [1,1], [3,4], [4,7], [5,6], [7,8], [6,0], [8,2]]
], dtype=int)



def edges2matrix(n_node, edges):
    N = n_node
    mat = np.zeros([N,N], dtype=int)
    mat[edges[:,1], edges[:,0]] = 1
    return mat

def block(route, state):
    N, _     = state.shape
    isfilled = np.sum(state, axis=1) > 0
    isstop   = np.sum(route[isfilled], axis=0) > 0
    croute   = np.copy(route)
    croute[:, isstop] = np.eye(N, dtype=np.int)[:, isstop]
    return croute

class Simulator:
    def setup_sample_network(self):
        N, M = sample_state.shape
        self.n_node = N
        self.n_spec = M
        self.edges  = None
        self.cand_edges  = sample_cand_edges
        self.cand_routes = np.array([ edges2matrix(N,es) for es in self.cand_edges])
        self.p_entry     = None
        self.reward_vec  = None
        self.state       = sample_state

    def step(self, action):
        now_state  = self.state
        route      = self.cand_routes[action]
        route      = block(route, now_state)
        next_state = np.dot(route, now_state)
        # next_state = add new box
        reward, terminal = None, None
        self.state = next_state
        return self.state, terminal, reward

    def make_graph(self, output='graph'):
        G = Digraph(format = 'png')
        for e in self.edges:
            G.edge(str(e[0]), str(e[1]))
        G.render(output)

if __name__ == '__main__':
    env = Simulator()
    env.setup_sample_network()
#    env.make_graph()
#    env.step([0,1])
