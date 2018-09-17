import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph

sample_N=11
sample_M=2
sample_edges = np.array([[0,1],[2,3],[1,4],[3,4],\
                         [4,5],[5,6],[6,7],[6,9],\
                         [7,8],[9,10],[8,0],[10,2]],dtype=int)
sample_state = np.zeros([sample_N, sample_M], dtype=int)
sample_state[5] = [1, 0]
sample_state[9] = [0, 1]
w0 = np.array([[0,0,0,0],[0,0,0,0],[1,1,0,0],[0,0,1,0]])
w1 = np.array([[0,1,0,0],[0,0,0,0],[1,0,0,0],[1,0,0,0]])

def gen_cuttedroutes(route):
    cutted_routes = None
    return cutted_routes

def block(route, state):
    """ OK
    : (in)route ... is NxN matrix.
    : (in)state ... is NxM matrix.
    : (out)croute ... is NxN matrix. cuted branch.
    """
    N, _     = net.shape
    isfilled = np.sum(route, axis=1) > 0
    isstop   = np.sum(route[isfilled], axis=0) > 0
    croute   = np.copy(route)
    croute[:, isstop] = np.eye(N, dtype=np.int)[:, isstop]
    return croute

def entry_box(now_state, next_state, p_entry):
    return None

def calc_reward(next_state, reward_vec):
    reward = None
    terminal = True
    return reward, terminal

class Simulator:
    def setup_sample_network(self):
        N = sample_N; M = sample_M
        self.n_node = N
        self.n_spec = M
        self.edges  = sample_edges
        self.route  = np.zeros([N, N], dtype=int)
        self.route[self.edges[:,1], self.edges[:,0]] = 1
        self.p_entry    = None
        self.reward_vec = None
        self.state      = sample_state
        self.cutted_routes = gen_cuttedroutes(self.route)

    def step(self, action):
        now_state = self.state
        croute = self.cutted_routes[action]
        croute = block(croute, now_state)
        next_state = np.dot(croute, now_state)
        next_state = entry_box(now_state, next_state, self.p_entry)
        reward, terminal = calc_reward(next_state, self.reward_vec)
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
    env.make_graph()
    env.step([0,1])
