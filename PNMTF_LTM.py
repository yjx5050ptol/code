import os
import time
import math

import numpy as np
from scipy import sparse
from mpi4py import MPI

from tqdm import tqdm
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import KG
import utils


metrics = ["c_v", "c_npmi", "c_uci", "u_mass"]
## Amazon
# label = ["Amazon", "Apps", "Automotive", "Baby", "Beauty", "Books", "CDs", "Cell", "Clothing", "Digital", "Electronics", "Grocery", "Health", "Home", "Kindle", "Movies", "Musical", "Office", "Patio", "Pet", "Sports", "Tools", "Toys", "Video"]
# AGNews
label = ["Business", "Entertainment", "Europe", "Health", "Sci", "Sports", "U", "World"]  # Sci: Sci & Tech;  U: U.S.A
## DBPedia
# label = ["Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace", "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"]


def tfidf(Dt_path):
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    docs = []
    y = []
    Dt_pt = open(Dt_path)
    X = vectorizer.fit_transform(Dt_pt)
    vocabulary = vectorizer.get_feature_names()
    tfidf = transformer.fit_transform(X)
    Dt_pt.close()

    y = []
    with open(Dt_path+'_label') as fobj:
        for line in fobj.readlines():
            label = line.strip()
            y.append(label)

    return tfidf.transpose(), vocabulary, y


def extract(W_t_T, top_n, vocabulary):
    num_word_topic = W_t_T.shape[0]
    E = []
    for i in range(num_word_topic):
        # 最大top_n个值的下标
        indices = np.argpartition(W_t_T[i], -top_n)[-top_n:].tolist()
        # topicword = [vocabulary[j] for j in indices].sort()
        topicword = [vocabulary[j] for j in indices]
        topicword.sort()    # 按字母排序
        E.append(topicword)
    return E


def adapt(vocabulary_tilde, W_tilde, vocabulary, num_word, fill_mode='rand'):
    """
    fill_model: 'rand' or 'zero' or 'initialization'
    """
    assert num_word == len(vocabulary)
    num_topic = W_tilde.shape[1]
    if fill_mode == 'rand':
        res = np.random.rand(num_word, num_topic).astype('float64')
    elif fill_mode == 'zero':
        res = np.zeros((num_word, num_topic), dtype='float64')
    elif fill_mode == 'initialization':
        res = np.random.rand(num_word, num_topic).astype('float64') * 0.1
        W_tilde = normalize(W_t, axis=0, norm='l2')  # axis 0: column; sum of a column will be 1 if 'l1' is used
    else:
        print('Unknown fill_mode')
        return None    
    i = 0
    j = 0
    while i < len(vocabulary_tilde) and j < len(vocabulary):
        if vocabulary_tilde[i] < vocabulary[j]:
            i += 1
        elif vocabulary_tilde[i] > vocabulary[j]:
            j += 1
        elif vocabulary_tilde[i] == vocabulary[j]:
            res[j] = W_tilde[i]
            i += 1
            j += 1
    return res


def texts_corpus_for_eval(pipeline):
    pipeline_texts = []
    pipeline_corpus = []
    for file_path in pipeline:
        file_pt = open(file_path, 'r')
        texts = []
        for line in file_pt:
            text = line.strip().split()
            texts.append(text)
        pipeline_texts.append(texts)
        pipeline_corpus.append(corpora.Dictionary(texts))
        file_pt.close()

    return pipeline_texts, pipeline_corpus


def find_top_word(W_t_T_sort_ind, top_n, vocabulary, pipeline_corpus):
    num_word_topic = W_t_T_sort_ind.shape[0]
    top_word = []
    pipeline_corpus_token = list(pipeline_corpus.token2id.keys())
    for i in range(num_word_topic):
        line = []
        for j in W_t_T_sort_ind[i]:
            if vocabulary[j] in pipeline_corpus_token: # word = vocabulary[j]
                line.append(vocabulary[j])
            if len(line) == top_n:
                break
        top_word.append(line)
    return top_word


def evaluate_coh(pipeline_texts, pipeline_corpus, W_t_T, top_n, vocabulary):
    ll_coh = [] # E_t在所有数据集上的coherence
    # 对W_t_T进行排序，返回下标。
    W_t_T_sort_ind = np.argsort(-W_t_T)
    for i in range(len(pipeline_texts)):
        # 获取的这20个词需要是在测coh的数据集上有的词，所以需要获取相应数据集上前20的词
        top_word = find_top_word(W_t_T_sort_ind, top_n, vocabulary, pipeline_corpus[i])
        coherence = {}
        methods = metrics
        for method in methods:
            coherence[method] = CoherenceModel(topics=top_word, texts=pipeline_texts[i], dictionary=pipeline_corpus[i], coherence=method).get_coherence()
        ll_coh.append(coherence)
    return ll_coh


def construct_mask(word_topic, top_n):
    # compute TU
    TU_list = utils.compute_TU_list(word_topic.T, top_n)
    M = np.array(TU_list, dtype='float64')
    print(np.average(M))
    M = np.zeros(np.shape(word_topic)) + M

    return M


def cal_blocksize(n, size, rank):
    if rank < (n % size):
        return int(math.ceil(n / size))
    else:
        return int(math.floor(n / size))


def summation(C_local, comm, rank=-1, counts=None, pattern='Allreduce'):
    """
    collective communication
    input a numpy array;
    pattern='Allreduce' or 'Reduce_scatter';
    rank and counts should be passed if 'Reduce_scatter' is passed.
    """
    # C_local = A_col.dot(B_row)
    if pattern == 'Allreduce':
        C = np.empty(C_local.shape, dtype='float64')
        comm.Allreduce([C_local, MPI.DOUBLE], [C, MPI.DOUBLE], op=MPI.SUM)
        return C
    elif pattern == 'Reduce_scatter':
        buffersize_p = counts[rank]
        colcount = C_local.shape[1]
        rowcount_p = buffersize_p // colcount
        C_row = np.empty((rowcount_p, colcount), dtype='float64')
        comm.Reduce_scatter([C_local, MPI.DOUBLE], [C_row, MPI.DOUBLE], recvcounts=counts, op=MPI.SUM)
        return C_row
    else:
        print('Unknown pattern!')
        return None


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    T = None
    max_step = None
    num_word_topic = None
    num_doc_topic = None
    lambda_kg = None
    lambda_tm = None
    lambda_c = None
    eps = None

    if comm_rank == 0:
        # 可期望pipeline由运行函数确定
        dataname = "AGNews"
        pipeline = ["agnews_{chunk_id}".format(chunk_id = chunk_id) for chunk_id in range(1,11)]
        
        T = len(pipeline)
        max_step = 200
        # max_step = 20   # for test
        num_word_topic = 20
        num_doc_topic = 20
        top_n = 20  # 主题词数目
        lambda_kg = 10
        lambda_tm = 1
        lambda_c = 1 # 0.0025
        eps = 1e-7
        # assert T == len(pipeline)

        print("max_step {max_step}, num_word_topic {num_word_topic}, num_doc_topic {num_doc_topic}, top_n {top_n}, "
              "lambda_kg {lambda_kg}, lambda_tm {lambda_tm}, lambda_c {lambda_c}, eps {eps}, p {p}.".format(
            max_step=max_step, num_word_topic=num_word_topic, num_doc_topic=num_doc_topic, top_n=top_n,
            lambda_kg=lambda_kg, lambda_tm=lambda_tm, lambda_c=lambda_c, eps=eps, p=comm_size) + "\n")

        if not os.path.exists("Log"):
            os.mkdir("Log")
        log_file = os.path.join("Log", "log.txt")
        topic_words_local_file = os.path.join("Log", "topic_words_local.txt")
        topic_words_global_file = os.path.join("Log", "topic_words_global.txt")
        
        log_files = [log_file, topic_words_local_file, topic_words_global_file]
        for f in log_files:
            with open(f, "a") as fobj:
                fobj.write("\n")
                fobj.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " PNMTF_LTM Start" + "\n")
                fobj.write(dataname + ": " + str(pipeline) + "\n")
                fobj.write("max_step {max_step}, num_word_topic {num_word_topic}, num_doc_topic {num_doc_topic}, "
                           "top_n {top_n}, lambda_kg {lambda_kg}, lambda_tm {lambda_tm}, lambda_c {lambda_c}, "
                           "eps {eps}, p {p}.".format(max_step=max_step, num_word_topic=num_word_topic,
                                               num_doc_topic=num_doc_topic, top_n=top_n, lambda_kg=lambda_kg,
                                               lambda_tm=lambda_tm, lambda_c=lambda_c, eps=eps, p=comm_size) + "\n")
        
        # for i in range(len(pipeline)):
        #     pipeline[i] = os.path.join(DATA_DIR, pipeline[i])
        pipeline = [os.path.join("Dataset", dataname, filename) for filename in pipeline]
        
        if not os.path.exists("Res"):
            os.mkdir("Res")
        result_path = os.path.join("Res", dataname)
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        
        # scores = NMTF_LTM(T, max_step, pipeline, num_word_topic, num_doc_topic, top_n, lambda_kg, lambda_tm, lambda_c, eps)
        # KG = ∅
        # KG graph()
        kg = KG.KG()
        W_old = None
        S_old = None
        vocabulary_tilde = None
        pipeline_texts, pipeline_corpus = texts_corpus_for_eval(pipeline)
        sgd_clf = SGDClassifier()  # for V
        sgd_clf_H = SGDClassifier()  # for H
        
        scores_W = []  # coherence for W
        scores_WS = []  # coherence for WS
        scores_clf = []  # classification scores
        scores_clf_H = []  # classification scores for H_t
    ## comm_rank 0 end
    
    T = comm.bcast(T, root=0)
    max_step = comm.bcast(max_step, root=0)
    num_word_topic = comm.bcast(num_word_topic, root=0)
    num_doc_topic = comm.bcast(num_doc_topic, root=0)
    lambda_kg = comm.bcast(lambda_kg, root=0)
    lambda_tm = comm.bcast(lambda_tm, root=0)
    lambda_c = comm.bcast(lambda_c, root=0)
    eps = comm.bcast(eps, root=0)

    for t in range(T):
        # pre-processing
        if comm_rank == 0:
            print("T =", t)
            # D_t
            Dt_path = pipeline[t]
            
            # preprocess Data_t to D_t tfidf
            # Sort D_t by word
            # get D_t sort word list
            D_t, vocabulary, y = tfidf(Dt_path)

            # 保证vocabulary有序
            # assert all(x <= y for x,y in zip(vocabulary, vocabulary[1:]))
            assert sorted(vocabulary) == vocabulary

            # D_tT = D_tT.toarray()
            D_t_colmajor = D_t.toarray()
            # D_t = D_t.transpose().toarray().T
            D_t = np.zeros_like(D_t_colmajor, order='C')
            D_t[:] = np.array(D_t_colmajor)

            num_word, num_doc = D_t.shape
            # num_word = len(vocabulary)
            print("num_word = ", num_word, "num_document = ", num_doc)

            # adapt W_(t-1) to D_t
            # 0时刻没有Consolidation项
            if t == 0:
                pass
            else:
                # caluculate W_tilde^(t-1) and M^(t-1)
                W_tilde = W_old.dot(S_old)
                M = construct_mask(W_tilde, top_n)
                # vocab adaptation
                W_tilde = adapt(vocabulary_tilde, W_tilde, vocabulary, num_word, fill_mode='zero')
                M = adapt(vocabulary_tilde, M, vocabulary, num_word, fill_mode='zero')
                #M = np.ones(W_tilde.shape)

            #Initialize S
            S_t = np.random.rand(num_word_topic, num_doc_topic).astype('float64')

            #Construct K_(t-1) from KG_(t-1)
            # 尺寸需要和Dt一致
            print('constructing K(t-1) ...')
            # K = np.zeros((num_word, num_word), dtype='float64')
            K = kg.construct(vocabulary, num_word)

            #diag(K_(t-1)1)
            print('constructing diagK ...')
            # diagK = np.zeros((num_word, num_word), dtype='float64')
            K_sum = K.sum(axis=1)
            # for i in range(num_word):
            #     diagK[i][i] = K_sum[i]
            diagK = np.diag(K_sum)
            print('construct diagK finish')

            # recv buffer
            V_t = np.empty((num_doc, num_doc_topic), dtype='float64')
            H_t = np.empty((num_doc, num_word_topic), dtype='float64')
            W_t = np.empty((num_word, num_word_topic), dtype='float64')
        else:  # comm_rank != 0
            num_word = None
            num_doc = None
            D_t = None
            D_t_colmajor = None
            W_tilde = None
            M = None
            W_t = None
            V_t = None
            H_t = None
            W_t = None
            K = None
            diagK = None
            S_t = np.empty((num_word_topic, num_doc_topic), dtype='float64')

        '''Parallelization handling'''
        # constants
        num_word = comm.bcast(num_word, root=0)
        num_doc = comm.bcast(num_doc, root=0)

        # buffer count
        num_word_p = cal_blocksize(num_word, comm_size, comm_rank)
        num_doc_p = cal_blocksize(num_doc, comm_size, comm_rank)

        counts_word_p = np.empty(comm_size, dtype='i')
        counts_doc_p = np.empty(comm_size, dtype='i')

        comm.Allgather(np.array([num_word_p], dtype='i'), counts_word_p)
        comm.Allgather(np.array([num_doc_p], dtype='i'), counts_doc_p)

        counts_wp_wt = counts_word_p * num_word_topic
        counts_wp_dt = counts_word_p * num_doc_topic
        counts_wp_w = counts_word_p * num_word
        counts_wp_d = counts_word_p * num_doc
        counts_dp_wt = counts_doc_p * num_word_topic
        counts_dp_dt = counts_doc_p * num_doc_topic
        counts_dp_w = counts_doc_p * num_word  # equivalent to w_dp
        
        displ_wp_wt = np.insert(np.cumsum(counts_wp_wt), 0, 0)[0:-1]  # np.cumsum: prefix sum
        displ_wp_dt = np.insert(np.cumsum(counts_wp_dt), 0, 0)[0:-1]
        displ_wp_w = np.insert(np.cumsum(counts_wp_w), 0, 0)[0:-1]
        displ_wp_d = np.insert(np.cumsum(counts_wp_d), 0, 0)[0:-1]
        displ_dp_wt = np.insert(np.cumsum(counts_dp_wt), 0, 0)[0:-1]
        displ_dp_dt = np.insert(np.cumsum(counts_dp_dt), 0, 0)[0:-1]
        displ_dp_w = np.insert(np.cumsum(counts_dp_w), 0, 0)[0:-1]
        
        # local blocks of matrices
        D_t_row = np.empty((num_word_p, num_doc), dtype='float64')
        D_t_col = np.empty((num_word, num_doc_p), dtype='float64', order='F')
        K_row = np.empty((num_word_p, num_word), dtype='float64')
        diagK_row = np.empty((num_word_p, num_word), dtype='float64')
        W_tilde_row = np.empty((num_word_p, num_doc_topic), dtype='float64')
        M_row = np.empty((num_word_p, num_doc_topic), dtype='float64')
        
        comm.Bcast(S_t, root=0)
        comm.Scatterv([D_t, counts_wp_d, displ_wp_d, MPI.DOUBLE], D_t_row, root=0)
        comm.Scatterv([D_t_colmajor, counts_dp_w, displ_dp_w, MPI.DOUBLE], D_t_col, root=0)
        comm.Scatterv([K, counts_wp_w, displ_wp_w, MPI.DOUBLE], K_row, root=0)
        comm.Scatterv([diagK, counts_wp_w, displ_wp_w, MPI.DOUBLE], diagK_row, root=0)
        K_rowT = K_row.T
        diagK_rowT = diagK_row.T
        if t != 0:
            comm.Scatterv([W_tilde, counts_wp_dt, displ_wp_dt, MPI.DOUBLE], W_tilde_row, root=0)
            comm.Scatterv([M, counts_wp_dt, displ_wp_dt, MPI.DOUBLE], M_row, root=0)
        
        D_t_row = sparse.csr_matrix(D_t_row)
        D_t_col = sparse.csr_matrix(D_t_col)
        K_rowT = sparse.csr_matrix(K_rowT)
        diagK_rowT = sparse.csr_matrix(diagK_rowT)
        
        # variable blocks initialization
        V_t_row = np.random.rand(num_doc_p, num_doc_topic).astype('float64')
        H_t_row = np.random.rand(num_doc_p, num_word_topic).astype('float64')
        W_t_row = np.random.rand(num_word_p, num_word_topic).astype('float64')
        
        '''Iterative Updates'''
        # Update W_t, S_t, V_t and H_t
        #if t != 0:
        #    M_W_tilde = M_row * W_tilde_row  # pre-calculate for W_t update
        for times in tqdm(range(max_step)):
            # update W_t
            SVT_col = S_t.dot(V_t_row.T)
            numerator_local = D_t_col.dot(SVT_col.T) + lambda_kg * K_rowT.dot(W_t_row) + lambda_tm * D_t_col.dot(H_t_row)
            numerator = summation(numerator_local, comm, rank=comm_rank, counts=counts_wp_wt, pattern='Reduce_scatter')
            temp_local = SVT_col.dot(SVT_col.T) + lambda_tm * H_t_row.T.dot(H_t_row)
            temp = summation(temp_local, comm, pattern='Allreduce')
            denominator = W_t_row.dot(temp)
            temp_local = diagK_rowT.dot(W_t_row)
            temp_row = summation(temp_local, comm, rank=comm_rank, counts=counts_wp_wt, pattern='Reduce_scatter')
            denominator += lambda_kg * temp_row
            #if t != 0:
            #    numerator += lambda_c * M_W_tilde.dot(S_t.T)
            #    denominator += lambda_c * (M_row * (W_t_row.dot(S_t))).dot(S_t.T)
            W_t_row = W_t_row * ((eps + numerator) / (eps + denominator))

            # update V_t
            WS_row = W_t_row.dot(S_t)
            temp_row = WS_row
            if t != 0:
                temp_row += lambda_c * W_tilde_row
            numerator_local = D_t_row.T.dot(temp_row)
            numerator = summation(numerator_local, comm, rank=comm_rank, counts=counts_dp_dt, pattern='Reduce_scatter')
            temp_local = WS_row.T.dot(WS_row)
            if t != 0:
                temp_local += lambda_c * W_tilde_row.T.dot(W_tilde_row)
            temp = summation(temp_local, comm, pattern='Allreduce')
            denominator = V_t_row.dot(temp)
            V_t_row = V_t_row * ((eps + numerator) / (eps + denominator))

            # update H_t
            numerator_local = D_t_row.T.dot(W_t_row)
            numerator = summation(numerator_local, comm, rank=comm_rank, counts=counts_dp_wt, pattern='Reduce_scatter')
            WTW_local = W_t_row.T.dot(W_t_row)
            WTW = summation(WTW_local, comm, pattern='Allreduce')
            denominator = H_t_row.dot(WTW)
            H_t_row = H_t_row * ((eps + numerator) / (eps + denominator))

            # update S_t
            temp_local = D_t_col.dot(V_t_row)
            temp_row = summation(temp_local, comm, rank=comm_rank, counts=counts_wp_dt, pattern='Reduce_scatter')
            #if t != 0:
            #    temp_row += lambda_c * M_W_tilde
            numerator_local = W_t_row.T.dot(temp_row)
            numerator = summation(numerator_local, comm, pattern='Allreduce')
            # WTW has been pre-calculated in the update of H_t.
            VTV_local = V_t_row.T.dot(V_t_row)
            VTV = summation(VTV_local, comm, pattern='Allreduce')
            denominator = WTW.dot(S_t).dot(VTV)
            #if t != 0:
            #    temp_local = W_t_row.T.dot(M_row * (W_t_row.dot(S_t)))
            #    temp = summation(temp_local, comm, pattern='Allreduce')
            #    denominator += lambda_c * temp
            S_t = S_t * ((eps + numerator) / (eps + denominator))

            # if comm_rank == 0 and comm_size == 1:
            #     print(np.linalg.norm(D_t - W_t_row.dot(S_t).dot(V_t_row.T)))  # only for comm_size = 1
            if (times+1) % 1000 == 0:
                print(times, comm_rank, W_t_row.shape, W_t_row)

        '''
        V_t_row = np.random.rand(num_doc_p, num_doc_topic).astype('float64')
        # post processing
        for times in tqdm(range(max_step)):
            # update V_t
            WS_row = W_t_row.dot(S_t)
            temp_row = WS_row
            #if t != 0:
            #    temp_row += lambda_c * W_tilde_row
            numerator_local = D_t_row.T.dot(temp_row)
            numerator = summation(numerator_local, comm, rank=comm_rank, counts=counts_dp_dt, pattern='Reduce_scatter')
            temp_local = WS_row.T.dot(WS_row)
            #if t != 0:
            #    temp_local += lambda_c * W_tilde_row.T.dot(W_tilde_row)
            temp = summation(temp_local, comm, pattern='Allreduce')
            denominator = V_t_row.dot(temp)
            V_t_row = V_t_row * ((eps + numerator) / (eps + denominator))
        '''
            
        '''Result Gathering'''
        comm.Gatherv(W_t_row, [W_t, counts_wp_wt, displ_wp_wt, MPI.DOUBLE], root=0)
        comm.Gatherv(V_t_row, [V_t, counts_dp_dt, displ_dp_dt, MPI.DOUBLE], root=0)
        comm.Gatherv(H_t_row, [H_t, counts_dp_wt, displ_dp_wt, MPI.DOUBLE], root=0)
        '''Evaluation'''
        if comm_rank == 0:
            
            utils.save_as_triple(W_t, os.path.join(result_path, 'W_'+str(t)+'.csv'))
            utils.save_as_triple(S_t, os.path.join(result_path, 'S_'+str(t)+'.csv'))
            utils.save_as_triple(V_t, os.path.join(result_path, 'V_'+str(t)+'.csv'))
            utils.save_as_triple(H_t, os.path.join(result_path, 'H_'+str(t)+'.csv'))
            
            print(W_t.dot(S_t))
            print(V_t.shape, V_t)
            # E_t = Extract(W_t)
            # E_t是主题词矩阵
            # E_t = extract(W_t.transpose(), top_n, vocabulary)
            print('extracting sorted topic words ...')
            E_t = extract(W_t.transpose(), 10, vocabulary)  # 用10个词更新KG
            # KG = KG + E_t
            print('updating KG ...')
            kg.update(E_t)

            # save W_t*S_t for next time as W_(t-1)*S_(t-1)
            W_old = W_t
            S_old = S_t
            vocabulary_tilde = vocabulary

            # evaluate coherence on all dataset
            # top_word = extract(W_t.transpose(), top_n, vocabulary)  #用20个词获得coherence
            # 问题是获取的这20个词需要是在测coh的数据集上有的词
            # top_word = find_top_word(W_t.transpose(), top_n, vocabulary)
            '''
            print('evaluating coherence ...')
            score_W = evaluate_coh(pipeline_texts, pipeline_corpus, W_t.transpose(), top_n, vocabulary)
            scores_W.append(score_W)
            print("coherence scores for W: \n", score_W)
            score_WS = evaluate_coh(pipeline_texts, pipeline_corpus, (W_t.dot(S_t)).transpose(), top_n, vocabulary)
            print("coherence scores for WS: \n", score_WS)
            scores_WS.append(score_WS)
            '''
            utils.print_topic_word(vocabulary, topic_words_local_file, W_t.T, top_n)
            utils.print_topic_word(vocabulary, topic_words_global_file, (W_t.dot(S_t)).T, top_n)

            # evaluate clustering, topic uniqueness, perplexity
            
            # evaluate classification
            # 随机80%训练，20%测试
            print('performing downstream classification ...')
            classes = np.array(label)
            # use V_t for classification
            X_train, X_test, y_train, y_test = train_test_split(V_t, y, test_size=0.2, random_state=666)
            sgd_clf.partial_fit(X_train, y_train, classes=label)
            y_pred = sgd_clf.predict(X_test)
            score_clf = accuracy_score(y_test, y_pred)
            scores_clf.append(score_clf)
            print("classification score using V:", score_clf)
            # use H_t for classification
            H_train, H_test, y_train, y_test = train_test_split(H_t, y, test_size=0.2, random_state=666)
            sgd_clf_H.partial_fit(H_train, y_train, classes=label)
            y_pred = sgd_clf_H.predict(H_test)
            score_clf_H = accuracy_score(y_test, y_pred)
            scores_clf_H.append(score_clf_H)
            print("classification score using H:", score_clf_H)
        ## comm_rank 0 end
        
    '''Write Log'''
    if comm_rank == 0:
        print(scores_W)
        print(scores_WS)
        print(scores_clf)
        print(scores_clf_H)
        
        with open(log_file, "a") as fobj:
            for metric in metrics:
                fobj.write("W: " + metric + "\n")
                for score in scores_W:
                    fobj.write(str([s[metric] for s in score]) + "\n")
                fobj.write("WS: " + metric + "\n")
                for score in scores_WS:
                    fobj.write(str([s[metric] for s in score]) + "\n")
            fobj.write("Classification using V: \n")
            fobj.write(str(scores_clf) + "\n")
            fobj.write("Classification using H: \n")
            fobj.write(str(scores_clf_H) + "\n")
            fobj.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " End" + "\n")
    ## comm_rank 0 end
