import numpy
from numpy.core.fromnumeric import sort
from sklearn.metrics.pairwise import cosine_similarity

class KG:
    def __init__(self):
        self.graph = {}

    def update(self, E_t):
        # E_t is a list, not an array
        num_word_topic = len(E_t)
        for i in range(num_word_topic):
            # 保证升序
            assert sorted(E_t[i]) == E_t[i]
            lis = E_t[i]
            for j in range(len(lis)):
                a = lis[j]
                # if not self.graph.has_key(a):
                if a not in self.graph:
                    self.graph[a] = {}
                for k in range(j + 1, len(lis)):
                    b = lis[k]
                    # if self.graph[a].has_key(b):
                    if b in self.graph[a]:
                        self.graph[a][b] += 1
                    else:
                        self.graph[a][b] = 1

    def construct(self, vocabulary, num_word):
        # Construct K_(t-1) from KG_(t-1)
        K = numpy.zeros((num_word, num_word), dtype='float64')
        # for evey pair(x, y) of vacabulary, K[x][y] = E(x, y) / max_(x,y)(E(x, y))
        maxedge = 1
        for i in range(num_word):
            # if self.graph.has_key(vocabulary[i]):
            if vocabulary[i] in self.graph:
                a = vocabulary[i]
                for j in range(i + 1, num_word):
                    # if self.graph[a].has_key(vocabulary[j]):
                    if vocabulary[j] in self.graph[a]:
                        b = vocabulary[j]
                        K[i][j] = K[j][i] = self.graph[a][b]
                        maxedge = max(maxedge, self.graph[a][b])

        K /= maxedge

        # 对角线元素为1
        for i in range(num_word):
            K[i][i] = 1

        return K

    def new_construct(self, word_idx, feature, vocabulary, num_word):
                # Construct K_(t-1) from KG_(t-1)
        print(feature.shape)
        print(num_word)
        #print(vocabulary)
        K = numpy.zeros((num_word, num_word), dtype='float64')
        # for evey pair(x, y) of vacabulary, K[x][y] = E(x, y) / max_(x,y)(E(x, y))
        maxedge = 1
        for i in range(num_word):
            # if self.graph.has_key(vocabulary[i]):
            if vocabulary[i] in word_idx:
                a = vocabulary[i]
                for j in range(i + 1, num_word):
                    # if self.graph[a].has_key(vocabulary[j]):
                    if vocabulary[j] in word_idx:
                        b = vocabulary[j]
                        K[i][j] = K[j][i] = cosine_similarity([feature[word_idx[a]],feature[word_idx[b]]])[0][1]
                        maxedge = max(maxedge, K[i][j])
        K /= maxedge
        return K
    
    def save_to_file(self):
        pass
    
    def graph_to_mat(self):
        idx = 0
        idx_word = {}
        word_idx = {}
        for word in self.graph:
            idx_word[idx] = word
            word_idx[word] = idx
            idx += 1
        adj = numpy.zeros((idx,idx),dtype="float64")
        for i in range(idx):
            for j in range(i + 1,idx):
                if idx_word[j] in self.graph[idx_word[i]]:
                    adj[i][j] += self.graph[idx_word[i]][idx_word[j]]
                adj[j][i] = adj[i][j]
        return word_idx, idx_word, adj