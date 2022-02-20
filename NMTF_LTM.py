import numpy
import os
import time

from tqdm import tqdm
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse

import KG
import utils


metrics = ["c_v", "c_npmi", "c_uci", "u_mass"]

def tfidf(Dt_path):
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    Dt_pt = open(Dt_path)
    X = vectorizer.fit_transform(Dt_pt)
    vocabulary = vectorizer.get_feature_names()
    tfidf = transformer.fit_transform(X)
    Dt_pt.close()
    return tfidf.transpose(), vocabulary

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

def construct_mask(word_topic, top_n):
    # compute TU
    TU_list = utils.compute_TU_list(word_topic.T, top_n)
    M = numpy.array(TU_list)
    print(numpy.average(M))
    M = numpy.zeros(numpy.shape(word_topic)) + M

    return M


def NMTF_LTM(T, max_step, pipeline, num_word_topic, num_doc_topic, top_n, lambda_kg, lambda_tm, lambda_c, eps):
    # KG = ∅
    # KG graph()
    kg = KG.KG()
    W_old = None
    S_old = None
    vocabulary_tilde = None
    pipeline_texts, pipeline_corpus = texts_corpus_for_eval(pipeline)
    
    scores = []

    for t in range(T):
        print("T =", t)
        # D_t
        Dt_path = pipeline[t]
        
        # preprocess Data_t to D_t tfidf
        # Sort D_t by word
        # get D_t sort word list
        D_t, vocabulary = tfidf(Dt_path)

        # 保证vocabulary有序
        # assert all(x <= y for x,y in zip(vocabulary, vocabulary[1:]))
        assert sorted(vocabulary) == vocabulary

        num_word, num_document = D_t.shape
        # num_word = len(vocabulary)
        print("num_word = ", num_word, "num_document = ", num_document)

        # adapt W_(t-1) to D_t
        # 0时刻没有Consolidation项
        if t == 0:
            pass
        else:
            W_adapt_old = adapt(vocabulary_tilde, W_old, vocabulary, num_word)
            W_tilde = W_adapt_old.dot(S_old)
            M = construct_mask(W_tilde, top_n)

        #Initialize W, S, V and H
        if t == 0:
            # if t==0, initialize W and S randomly
            W_t = numpy.random.rand(num_word, num_word_topic).astype('float64')
            S_t = numpy.random.rand(num_word_topic, num_doc_topic).astype('float64')
        else:
            # if t != 0, initialize W and S with W_(t-1) and S_(t-1)
            W_t = W_adapt_old
            S_t = S_old
        V_t = numpy.random.rand(num_document, num_doc_topic).astype('float64')
        H_t = numpy.random.rand(num_document, num_word_topic).astype('float64')

        #Construct K_(t-1) from KG_(t-1)
        # 尺寸需要和Dt一致
        print('constructing K(t-1) ...')
        # K = numpy.zeros((num_word, num_word), dtype='float64')
        K = kg.construct(vocabulary, num_word)

        #diag(K_(t-1)1)
        print('constructing diagK ...')
        # diagK = numpy.zeros((num_word, num_word), dtype='float64')
        K_sum = K.sum(axis=1)
        # for i in range(num_word):
        #     diagK[i][i] = K_sum[i]
        diagK = numpy.diag(K_sum)

        K = sparse.csc_matrix(K)
        diagK = sparse.csc_matrix(diagK)

        print('construct diagK finish')

        # Update W_t, S_t, V_t and H_t
        for times in tqdm(range(max_step)):
            # update W_t
            SVT = S_t.dot(V_t.T)
            numerator = D_t.dot(SVT.T) + lambda_kg * K.dot(W_t) + lambda_tm * D_t.dot(H_t)
            denominator = W_t.dot(SVT.dot(SVT.T)) + lambda_kg * diagK.dot(W_t) + lambda_tm * W_t.dot(H_t.T.dot(H_t))
            if t != 0:
                numerator += lambda_c * (M * W_tilde).dot(S_t.T)
                denominator += lambda_c * (M * (W_t.dot(S_t))).dot(S_t.T)
            W_t = W_t * ((eps + numerator) / (eps + denominator))

            # update V_t
            WS = W_t.dot(S_t)
            numerator = D_t.T.dot(WS)
            denominator = V_t.dot(WS.T.dot(WS))
            V_t = V_t * ((eps + numerator) / (eps + denominator))

            # update H_t
            numerator = D_t.T.dot(W_t)
            denominator = H_t.dot(W_t.T.dot(W_t))
            H_t = H_t * ((eps + numerator) / (eps + denominator))

            # update S_t
            numerator = W_t.T.dot(D_t.dot(V_t))
            denominator = W_t.T.dot(W_t).dot(S_t).dot(V_t.T.dot(V_t))
            if t != 0:
                numerator += lambda_c * W_t.T.dot(M * W_tilde)
                denominator += lambda_c * W_t.T.dot(M * (W_t.dot(S_t)))
            S_t = S_t * ((eps + numerator) / (eps + denominator))
            print(numpy.linalg.norm(D_t - W_t.dot(S_t).dot(V_t.T)))

        # E_t = Extract(W_t)
        # E_t是主题词矩阵
        # E_t = extract(W_t.transpose(), top_n, vocabulary)
        E_t = extract(W_t.transpose(), 10, vocabulary)  # 用10个词更新KG
        # KG = KG + E_t
        kg.update(E_t)

        # save W_t*S_t for next time as W_(t-1)*S_(t-1)
        W_old = W_t
        S_old = S_t
        vocabulary_tilde = vocabulary

        # evaluate coherence on all dataset
        # top_word = extract(W_t.transpose(), top_n, vocabulary)  #用20个词获得coherence
        # 问题是获取的这20个词需要是在测coh的数据集上有的词
        # top_word = find_top_word(W_t.transpose(), top_n, vocabulary)

        score = evaluate_coh(pipeline_texts, pipeline_corpus, (W_t.dot(S_t)).transpose(), top_n, vocabulary)

        print("score", score)
        scores.append(score)

        # evaluate clustering, topic uniqueness, perplexity
        
    return scores


if __name__ == '__main__':
    if not os.path.exists("Log"):
        os.mkdir("Log")
    log_file = os.path.join("Log", "log.txt")

    # 可期望pipeline由运行函数确定
    pipeline = ["20news_min_cnt.txt", "classic4_min_cnt.txt", "stackoverflow_min_cnt2.txt", "webkb_min_cnt2.txt"]
    # pipeline = ["classic4_min_cnt.txt", "stackoverflow_min_cnt2.txt", "webkb_min_cnt2.txt"]
    
    # T = 4
    # T = 3   # for test
    T = len(pipeline)

    max_step = 200
    # max_step = 20   # for test
    num_word_topic = 50
    num_doc_topic = 100
    top_n = 20  # 主题词数目

    lambda_kg = 10000
    lambda_tm = 1
    lambda_c = 10000

    eps = 1e-7
    # assert T == len(pipeline)
    
    with open(log_file, "a") as fobj:
        fobj.write("\n")
        fobj.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " NMTF_LTM Start" + "\n")
        fobj.write(str(pipeline) + "\n")
        fobj.write("max_step {max_step}, num_word_topic {num_word_topic}, num_doc_topic {num_doc_topic}, top_n {top_n}, lambda_kg {lambda_kg}, lambda_tm {lambda_tm}, lambda_c {lambda_c}, eps {eps}.".format(max_step=max_step, num_word_topic=num_word_topic, num_doc_topic=num_doc_topic, top_n=top_n, lambda_kg=lambda_kg, lambda_tm=lambda_tm, lambda_c=lambda_c, eps=eps) + "\n")
    
    # for i in range(len(pipeline)):
    #     pipeline[i] = os.path.join(DATA_DIR, pipeline[i])
    pipeline = [os.path.join("Dataset", filename) for filename in pipeline]
        
    scores = NMTF_LTM(T, max_step, pipeline, num_word_topic, num_doc_topic, top_n, lambda_kg, lambda_tm, lambda_c, eps)
    print(scores)
    
    with open(log_file, "a") as fobj:
        for metric in metrics:
            fobj.write(metric + "\n")
            for score in scores:
                fobj.write(str([s[metric] for s in score]) + "\n")
        fobj.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " End" + "\n")
