import time
import numpy
import os

from tqdm import tqdm
from scipy import sparse
from scipy.sparse import construct
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import KG
import utils

from spectral import affinity, clustering

metrics = ["c_v", "c_npmi", "c_uci", "u_mass"]
# AGNews
label = ["Business", "Entertainment", "Europe", "Health", "Sci", "Sports", "U", "World"]  # Sci: Sci & Tech;  U: U.S.A


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
        indices = numpy.argpartition(W_t_T[i], -top_n)[-top_n:].tolist()
        # topicword = [vocabulary[j] for j in indices].sort()
        topicword = [vocabulary[j] for j in indices]
        topicword.sort()    # 按字母排序
        E.append(topicword)
    return E

'''
def adapt(vocabulary_tilde, W_tilde, vocabulary, num_word):
    assert num_word == len(vocabulary)
    num_topic = W_tilde.shape[1]
    res = numpy.zeros((num_word, num_topic), dtype='float64')
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
'''

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
            if len(line) == 20:
                break
        top_word.append(line)
    return top_word


def evaluate_coh(pipeline_texts, pipeline_corpus, W_t_T, top_n, vocabulary):
    ll_coh = [] # E_t在所有数据集上的coherence
    # 对W_t_T进行排序，返回下标。
    W_t_T_sort_ind = numpy.argsort(-W_t_T)
    for i in range(len(pipeline_texts)):
        # 获取的这20个词需要是在测coh的数据集上有的词，所以需要获取相应数据集上前20的词
        top_word = find_top_word(W_t_T_sort_ind, top_n, vocabulary, pipeline_corpus[i])
        coherence = {}
        methods = metrics
        for method in methods:
            coherence[method] = CoherenceModel(topics=top_word, texts=pipeline_texts[i], dictionary=pipeline_corpus[i], coherence=method).get_coherence()
        ll_coh.append(coherence)
    return ll_coh


def construct_C(path):
    label_npy = path + r"_C.csv"
    if os.path.exists(label_npy):
        C = utils.read_triple(label_npy)
        return C
    # else
    label_path = path + r"_label"
    print(label_path)
    label_pt = open(label_path, 'r')
    labels = label_pt.readlines()
    num_doc = len(labels)
    C = numpy.zeros((num_doc, num_doc), dtype='float64')
    start = 0
    while start < num_doc:
        end = start
        while end + 1 < num_doc and labels[end + 1] == labels[start]:
            end += 1
        for i in range(start, end + 1):
            for j in range(start, end + 1):
                C[i][j] = 1.0
        start = end + 1
    # if :
    #     end += 1
    utils.save_as_triple(C, label_npy)
    return C


def NMF_LTM(T, max_step, pipeline, num_word_topic, top_n, alpha, beta, gamma, lambda_, eps, dataname):
    
    if not os.path.exists("Log_NMF-LTM"):
        os.mkdir("Log_NMF-LTM")
    log_file = os.path.join("Log_NMF-LTM", "log.txt")
    topic_words_file = os.path.join("Log_NMF-LTM", "topic_words.txt")
    
    log_files = [log_file, topic_words_file]
    for f in log_files:
        with open(f, "a") as fobj:
            fobj.write("\n")
            fobj.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " NMF-LTM Start" + "\n")
            fobj.write(dataname + "\n")
            fobj.write("max_step {max_step}, num_word_topic {num_word_topic}, top_n {top_n}, "
                        "alpha {alpha}, beta {beta}, gamma {gamma}, lambda_ {lambda_}, eps {eps}.".format(
                        max_step=max_step, num_word_topic=num_word_topic, top_n=top_n,
                        alpha=alpha, beta=beta, gamma=gamma, lambda_=lambda_, eps=eps) + "\n")
    
    if not os.path.exists("Res_NMF-LTM"):
        os.mkdir("Res_NMF-LTM")
    result_path = os.path.join("Res_NMF-LTM", dataname)
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    # KG = ∅
    # KG graph()
    kg = KG.KG()
    # W_old = None
    # S_old = None
    # vocabulary_tilde = None
    pipeline_texts, pipeline_corpus = texts_corpus_for_eval(pipeline)
    
    sgd_clf = SGDClassifier()
    
    scores_topic = []  # coherence for U
    scores_clf = []  # classification scores

    for t in range(T):
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

        num_word, num_document = D_t.shape
        # num_word = len(vocabulary)
        print("num_word = ", num_word, "num_document = ", num_document)

        U = numpy.random.rand(num_word, num_word_topic).astype('float64')
        V = numpy.random.rand(num_word_topic, num_document).astype('float64')

        #Construct W_(t-1) from KG_(t-1)
        # 尺寸需要和Dt一致
        print('constructing W(t-1) ...')
        # K = numpy.zeros((num_word, num_word), dtype='float64')
        word_idx, idx_word, adj = kg.graph_to_mat()
        print(adj.shape)
        if(adj.shape[0] > 0):
            feature = clustering.spectral_clustering(adj, 6)
            W = kg.new_construct(word_idx, feature, vocabulary, num_word)
        else:
            W = kg.construct(vocabulary, num_word)

        #diag(K_(t-1)1)
        print('constructing diagW ...')
        # diagK = numpy.zeros((num_word, num_word), dtype='float64')
        W_sum = W.sum(axis=1)
        # for i in range(num_word):
        #     diagK[i][i] = K_sum[i]
        diagW = numpy.diag(W_sum)

        W = sparse.csr_matrix(W)
        diagW = sparse.csr_matrix(diagW)
        print('construct diagW finish')

        # Construct C from data
        C = construct_C(pipeline[t])
        # diagC
        C_sum = C.sum(axis=1)
        diagC = numpy.diag(C_sum)
        
        C = sparse.csr_matrix(C)
        diagC = sparse.csr_matrix(diagC)
        print('construct diagC finish')

        # Update W_t, S_t, V_t and H_t
        for times in tqdm(range(max_step)):
            # update U
            numerator = D_t.dot(V.T) + alpha * W.dot(U) + 2 * beta * U
            denominator = U.dot(V.dot(V.T)) + alpha * diagW.dot(U) + 2 * beta * U.dot(U.T).dot(U)
            U = U * ((numerator + eps) / (denominator + eps))
            # update V
            numerator = (D_t.T.dot(U)).T  # to exploit csr: (D_t.T.dot(U)).T<==> U.T.dot(D_t.toarray())
            denominator = U.T.dot(U).dot(V) + 0.5 * lambda_ * numpy.ones_like(V)
            if gamma != 0:
                numerator += gamma * (C.T.dot(V.T)).T  # to exploit csr: (C.T.dot(V.T)).T <==> V.dot(C.toarray())
                denominator += gamma * (diagC.T.dot(V.T)).T  # to exploit csr: (diagC.T.dot(V.T)).T <==> V.dot(diagC.toarray())
            V = V * ((numerator + eps) / (denominator + eps))

        
        utils.save_as_triple(U, os.path.join(result_path, 'U_'+str(t)+'.csv'))
        utils.save_as_triple(V, os.path.join(result_path, 'V_'+str(t)+'.csv'))
        
        # E_t = Extract(W_t)
        # E_t是主题词矩阵
        # E_t = extract(W_t.transpose(), top_n, vocabulary)
        E_t = extract(U.transpose(), 10, vocabulary)  # 用10个词更新KG
        # KG = KG + E_t
        kg.update(E_t)

        # evaluate coherence on all dataset
        # top_word = extract(W_t.transpose(), top_n, vocabulary)  #用20个词获得coherence
        # 问题是获取的这20个词需要是在测coh的数据集上有的词
        # top_word = find_top_word(W_t.transpose(), top_n, vocabulary)
        score = evaluate_coh(pipeline_texts, pipeline_corpus, U.transpose(), top_n, vocabulary)
        scores_topic.append(score)
        print("score", score)
        
        utils.print_topic_word(vocabulary, topic_words_file, U.T, top_n)

        # evaluate clustering, topic uniqueness, perplexity
        
        # evaluate classification
        # 随机80%训练，20%测试
        print('performing downstream classification ...')
        classes = numpy.array(label)
        # use V_t for classifications
        X_train, X_test, y_train, y_test = train_test_split(V.T, y, test_size=0.2, random_state=666)
        sgd_clf.partial_fit(X_train, y_train, classes=label)
        y_pred = sgd_clf.predict(X_test)
        score_clf = accuracy_score(y_test, y_pred)
        scores_clf.append(score_clf)
        print("classification score:", score_clf)
        
    # final results    
    print(scores_topic)
    print(scores_clf)
    
    with open(log_file, "a") as fobj:
        for metric in metrics:
            fobj.write("U: " + metric + "\n")
            for score in scores_topic:
                fobj.write(str([s[metric] for s in score]) + "\n")
        fobj.write("Classification using V: \n")
        fobj.write(str(scores_clf) + "\n")
        fobj.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " End" + "\n")


if __name__ == '__main__':

    # 可期望pipeline由运行函数确定
    dataname = "20News"
    pipeline = ["20news_{chunk_id}".format(chunk_id = chunk_id) for chunk_id in range(1,6)]
    
    pipeline = [os.path.join("../Dataset", dataname, filename) for filename in pipeline]
    
    T = len(pipeline)
    max_step = 200
    # max_step = 20   # for test
    num_word_topic = 20
    top_n = 20  # 主题词数目
    alpha = 10
    beta = 0 #0.5
    gamma = 0 #0.001
    lambda_ = 0 #0.001
    eps = 1e-7
    
    print("max_step {max_step}, num_word_topic {num_word_topic}, top_n {top_n}, "
              "alpha {alpha}, beta {beta}, gamma {gamma}, lambda_ {lambda_}, eps {eps}.".format(
            max_step=max_step, num_word_topic=num_word_topic, top_n=top_n,
            alpha=alpha, beta=beta, gamma=gamma, lambda_=lambda_, eps=eps) + "\n")

    NMF_LTM(T, max_step, pipeline, num_word_topic, top_n, alpha, beta, gamma, lambda_, eps, dataname)
