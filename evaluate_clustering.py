import os
import pandas as pd
import numpy as np
from scipy import sparse

from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
# from sklearn.cluster import KMeans
from coclust.clustering import SphericalKmeans

data_dir = r'Dataset/AGNews'
#res_dir = r'Res/AGNews'
res_dir = r'Res/AGNews_lambC_1'
chunknum = 10
label_file_list = [os.path.join(data_dir, 'agnews_'+str(i+1)+'_label') for i in range(chunknum)]
res_file_list = [os.path.join(res_dir, 'V_'+str(i)+'.csv') for i in range(chunknum)]
n_clusters = 8  # the number of ground-truth class labels
mode = 'doc_topic'  # topic_doc


def read_triple(filename):
    tp = pd.read_csv(open(filename))
    rows, cols, data = np.array(tp['row_idx']), np.array(tp['col_idx']), np.array(tp['data'])
    return sparse.coo_matrix((data, (rows, cols)), shape=(max(rows)+1, max(cols)+1)).toarray()


def get_true_label(filename_list):
    res = []
    label_id_mapping = {}
    label_id_count = 0
    
    for filename in filename_list:
        with open(filename) as fobj:
            for line in fobj.readlines():
                label = line.strip()
                if label in label_id_mapping:
                    res.append(label_id_mapping[label])
                else:
                    res.append(label_id_count)
                    label_id_mapping[label] = label_id_count
                    label_id_count += 1
    return np.array(res)


def cal_indexes(labels_true, labels_pred):
    """
    RI and NMI
    """
    ri = rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred) #average_method='arithmetic'
    return nmi, ri


if __name__ == '__main__':
    input_matrices = []
    for filename in res_file_list:
        input_matrices.append(read_triple(filename))
    if mode == 'doc_topic':
        X = np.concatenate(input_matrices, axis=0)
        X = sparse.csr_matrix(X)
    elif mode == 'topic_doc':
        X = np.concatenate(input_matrices, axis=1)
        X = sparse.csr_matrix(X.T)
    print(X.shape)
    
    model = SphericalKmeans(n_clusters=n_clusters, max_iter=20, weighting=False)  # weighting: perform TF-IDF inside
    model.fit(X)
    labels_pred = model.labels_
    
    labels_true = get_true_label(label_file_list)
    nmi, ri = cal_indexes(labels_true, labels_pred)
    
    print('NMI:',nmi, 'RI:', ri)
