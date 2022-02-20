import os
import numpy as np
import pandas as pd
from scipy import sparse
import sklearn.metrics.pairwise as pw

import time
from tqdm import tqdm

# data_dir = r'F:\learn\Master\Research-areas\NMF\LTMNMTF\nmtf-ltm\Information Retrieval\Data\AGNews'
# res_dir = r'F:\learn\Master\Research-areas\NMF\LTMNMTF\nmtf-ltm\Res_NMTF-LTM\AGNews_lambC_1'
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset", "AGNews")
res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "Res", "AGNews_lambC_1")  # Res_NMF
chunknum = 10
printAsc = False    # 是否输出时间信息
recall = [5, 10, 0.02]
sample = False

def get_label(filename):
    with open(filename) as fobj:
        return fobj.read().splitlines()

def read_triple(filename):
    tp = pd.read_csv(open(filename))
    rows, cols, data = np.array(tp['row_idx']), np.array(tp['col_idx']), np.array(tp['data'])
    return sparse.coo_matrix((data, (rows, cols)), shape=(max(rows)+1, max(cols)+1)).toarray()

def compare_labels(train_labels, test_label, label_type="", evaluation_type="", labels_to_count=[]):
    #train_labels = train_labels[:, labels_to_count]
    #test_label = test_label[labels_to_count]

    vec_goodLabel = []

    # train_labels = np.asarray(train_labels) # 将数据转化为ndarray

    if label_type == "single":
        # if not (isinstance(test_label, int) or isinstance(train_labels[0], int)):
        #     print("Labels are not instances of int")
        #     exit()

        # test_labels = np.ones(train_labels.shape[0], dtype=np.float32) * test_label
        test_labels = [test_label] * train_labels.shape[0]
        # test_label是标量，一个值，星乘1向量得到全test_label向量

        vec_goodLabel = np.array((train_labels == test_labels), dtype=np.int8)
    elif label_type == "multi":
        if not len(train_labels[0]) == len(test_label):
            print("Mismatched label vector length")
            exit()

        test_labels = np.asarray(test_label)
        labels_comparison_vec = np.dot(train_labels, test_labels)

        if evaluation_type == "relaxed":
            vec_goodLabel = np.array((labels_comparison_vec != 0), dtype=np.int8)

        elif evaluation_type == "strict":
            test_label_vec = np.ones(train_labels.shape[0]) * np.sum(test_label)
            vec_goodLabel = np.array((labels_comparison_vec == test_label_vec), dtype=np.int8)

        else:
            print("Invalid evaluation_type value.")

    else:
        print("Invalid label_type value.")

    return vec_goodLabel

def perform_IR_prec(kernel_matrix_test, train_labels, test_labels, list_percRetrieval=None, single_precision=False, label_type="", evaluation="", index2label_dict=None, labels_to_not_count=[], corpus_docs=None, query_docs=None, IR_filename=""):
    '''
    :param kernel_matrix_test: shape: size = |test_samples| x |train_samples|                   两个矩阵的余弦相似度矩阵，cosine_similarity(corpus_vectors, query_vectors).T
    :param train_labels:              size = |train_samples| or |train_samples| x num_labels
    :param test_labels:               size = |test_samples| or |test_samples| x num_labels
    :param list_percRetrieval:        list of fractions or number at which IR has to be calculated
    :param single_precision:          True, if only one fraction is used
    :param label_type:                "single" or "multi"
    :param evaluation:                "strict" or "relaxed", only for 
    :return:
    '''
    #print('Computing IR prec......')

    if not len(test_labels) == len(kernel_matrix_test):
        print('mismatched samples in test_labels and kernel_matrix_test')
        exit()
        # kernel_matrix_test: q x c, test_label: q, train_label: c

    labels_to_count = []
    #if labels_to_not_count:
    #    for index, label in index2label_dict.iteritems():
    #        if not label in labels_to_not_count:
    #            labels_to_count.append(int(index))

    prec = []

    if single_precision:    # recall只有一个数
        vec_simIndexSorted = np.argsort(kernel_matrix_test, axis=1)[:, ::-1]
        # 返回对行内进行数组值排序后的索引值，[:,::-1]表示倒序，即行都是数组值降序的索引值
        # 行代表train的部分
        prec_num_docs = np.floor(list_percRetrieval[0] * kernel_matrix_test.shape[1])   # R的数量
        vec_simIndexSorted_prec = vec_simIndexSorted[:, :int(prec_num_docs)]    # 每个test对应的前R个train的下标
        
        for counter, indices in enumerate(vec_simIndexSorted_prec):
            if label_type == "multi":
                classQuery = test_labels[counter, :]
                tr_labels = train_labels[indices, :]
            else:
                classQuery = test_labels[counter]
                tr_labels = train_labels[indices]
            list_percPrecision = np.zeros(len(list_percRetrieval))  #?

            vec_goodLabel = compare_labels(tr_labels, classQuery, label_type=label_type)

            list_percPrecision[0] = np.sum(vec_goodLabel) / float(len(vec_goodLabel))

            prec += [list_percPrecision]
    else:   # recall有多个数
        list_totalRetrievalCount = []   #list of number at which IR has to be calculated
        for frac in list_percRetrieval:
            if frac < 1:    # 小于1是比例
                list_totalRetrievalCount.append(int(np.floor(frac * kernel_matrix_test.shape[1])))
            else:   # 大于1是数量
                list_totalRetrievalCount.append(frac)
            # frac * train的数量
        if sorted(list_totalRetrievalCount) != list_totalRetrievalCount:
            print("recall is not ascending.")
            exit()
        start = time.time()
        # vec_simIndexSorted = np.argsort(kernel_matrix_test, axis=1)[:, ::-1]
        vec_simIndexSorted = np.argsort(kernel_matrix_test, axis=1)[:, :-list_totalRetrievalCount[-1]-1:-1] # 最大降序K个值的索引
        end = time.time()
        print("argsort time:", end-start)
        print("vec_simIndexSorted", time.asctime( time.localtime(time.time()) ))
        # 返回对行内进行数组值排序后的索引值，[:,::-1]表示倒序，即行都是数组值降序的索引值
        
        # 行代表train的部分
        for counter, indices in tqdm(enumerate(vec_simIndexSorted)):
        # for counter, indices in enumerate(vec_simIndexSorted):
            if label_type == "multi":
                classQuery = test_labels[counter, :]
                tr_labels = train_labels[indices, :]    # tr_labels是与query cos相似度降序的的train
            else:
                classQuery = test_labels[counter]
                tr_labels = np.array(train_labels)[indices]
            if printAsc: print("choose label", time.asctime( time.localtime(time.time()) ))
            
            # list_percRetrieval: list of fractions at which IR has to be calculated
            vec_goodLabel = compare_labels(tr_labels, classQuery, label_type=label_type, evaluation_type=evaluation, labels_to_count=labels_to_count)
            if printAsc: print("compare_labels", time.asctime( time.localtime(time.time()) ))
            
            countGoodLabel = 0
            list_percPrecision = np.zeros(len(list_percRetrieval))  # 不能放在循环外面初始化，这会导致prec都是同一个数组
            for indexRetrieval, totalRetrievalCount in enumerate(list_totalRetrievalCount):
                if indexRetrieval == 0:
                    countGoodLabel += np.sum(vec_goodLabel[:int(totalRetrievalCount)])
                else:
                    countGoodLabel += np.sum(vec_goodLabel[int(lastTotalRetrievalCount):int(totalRetrievalCount)])

                list_percPrecision[indexRetrieval] = countGoodLabel / float(totalRetrievalCount)
                lastTotalRetrievalCount = totalRetrievalCount
                # list_percPrecision[indexRetrieval] = np.sum(vec_goodLabel[:totalRetrievalCount])    # 一句解决
            
            if printAsc: print("count", time.asctime( time.localtime(time.time()) ))

            # prec += [list_percPrecision]  # vec_simIndexSorted[:int(list_totalRetrievalCount[0])]
            prec.append(list_percPrecision)  # vec_simIndexSorted[:int(list_totalRetrievalCount[0])]
            """
            with open(IR_filename, "a") as f:
                f.write("Query\t::\t" + query_docs[counter] + "\n\n")
                for index in indices[:int(kernel_matrix_test.shape[1] * 0.02)]:
                    f.write(str(kernel_matrix_test[counter, index]) + "\t::\t" + corpus_docs[index] + "\n")
                f.write("\n\n")
            """
        print("compare and count", time.asctime( time.localtime(time.time()) ))
            

    prec = np.mean(prec, axis=0)
    # print('prec:', prec)
    return prec

if __name__ == '__main__':

    label_file_list = [os.path.join(data_dir, 'agnews_'+str(i+1)+'_label') for i in range(chunknum)]
    res_file_list = [os.path.join(res_dir, 'V_'+str(i)+'.csv') for i in range(chunknum)]
    label_list = []

    for filename in label_file_list:
        label_list.append(get_label(filename))

    doc_topic_matrices = []
    for filename in res_file_list:
        doc_topic_matrices.append(read_triple(filename))
    rb = [] # 最终结果
    for i in range(1, chunknum):
        query_vectors = doc_topic_matrices[i]
        if sample: query_vectors = query_vectors[::20]
        query_label = label_list[i]
        if sample: query_label = query_label[::20]
        ra = []
        for j in range(i):
            corpus_vectors = doc_topic_matrices[j]
            corpus_label = label_list[j]
            single_precision = False if len(recall) > 1 else True
            start = time.time()
            similarity_matrix = pw.cosine_similarity(query_vectors, corpus_vectors)
            end = time.time()
            print("similrity time, query:%d, train:%d, time:%f" %(i, j, end-start))
            print("similarity_matrix", time.asctime( time.localtime(time.time()) ))
            results = perform_IR_prec(similarity_matrix, corpus_label, query_label, list_percRetrieval=recall, single_precision=single_precision, label_type="single")
            # ra = [[results[k]] for k in range(len(recall))] if i == 0 else [ra[k].append(results[k]) for k in range(len(recall))]
            if j == 0:
                ra = [[results[k]] for k in range(len(recall))]
            else:
                for k in range(len(recall)):
                    ra[k].append(results[k])
        # rb = [[ra[k]] for k in range(len(ra))] if j == 1 else [rb[k].append(ra[k]) for k in range(len(ra))]
        if i == 1:
            rb = [[ra[k]] for k in range(len(ra))]
        else:
            for k in range(len(recall)):
                rb[k].append(ra[k])
        print(rb)
    