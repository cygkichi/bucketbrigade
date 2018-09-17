import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph

def branch_cut(w, cut_matrixs):
    b, N, _ = cut_matrixs.shape
    wc = np.copy(w)
    for cut_mat in cut_matrixs:
        wc = wc * cut_mat
    return wc

def anti_collision(wc, now_state):
    N, _ = wc.shape
    isfilled = np.sum(now_state, axis=1) > 0
    isstop = np.sum(wc[isfilled], axis=0) > 0
    ws = np.copy(wc)
    ws[:, isstop] = np.eye(N, dtype=np.int)[:, isstop]
    return ws

def entry_box(now_state, next_state, p_entry):
    return None

def calc_reward(next_state, reward_vec):
    return None


class Simulator:
    def setup_sample_network(self):
        N = 11; M = 2
        self.n_node = N
        self.n_spec = M
        self.edges = np.array([[0,1],[2,3],[1,4],[3,4],[4,5],\
                               [5,6],[6,7],[6,9],[7,8],[9,10],[8,0],[10,2]],dtype=int)
        self.w = np.zeros([N, N], dtype=int)
        self.w[self.edges[:,1], self.edges[:,0]] = 1
        self.p_entry = None
        self.reward_vec = None
        self.state = np.zeros([N,M], dtype=int)
        self.state[5] = [1,0]
        self.state[9] = [0,1]
        cut_matrixs =[]
        for i, hline in enumerate(self.w):
            if np.sum(hline) > 1:
                mats = []
                for j,v in enumerate(hline):
                    if v > 0:
                        mat = np.ones([N,N], dtype=int)
                        mat[i] = 0
                        mat[i,j] = 1
                        mats.append(mat)
                cut_matrixs.append(mats)
        for j, vline in enumerate(self.w.T):
            if np.sum(vline) > 1:
                mats = []
                for i,v in enumerate(vline):
                    if v > 0:
                        mat = np.ones([N,N], dtype=int)
                        mat[:,j] = 0
                        mat[i,j] = 1
                        mats.append(mat)
                cut_matrixs.append(mats)
        self.cut_matrix_set = np.array(cut_matrixs, dtype=np.int)

    def next(self, action):
        now_state = self.state
        cut_matrixs = self.cut_matrix_set[[0,1],action]
        self.wc = branch_cut(self.w, cut_matrixs)
        self.ws = anti_collision(self.wc, now_state)
        next_state = np.dot(self.ws, now_state)
        #next_state = entry_box(now_state, next_state, self.p_entry)
        #reward = calc_reward(next_state, self.reward_vec)
        self.state = next_state
        terminal = True
#        return self.state, terminal, reward

    def make_graph(self, output='graph'):
        G = Digraph(format = 'png')
        for e in self.edges:
            G.edge(str(e[0]), str(e[1]))
        G.render(output)

if __name__ == '__main__':
    env = Simulator()
    env.setup_sample_network()
    env.make_graph()
    env.next([0,1])
