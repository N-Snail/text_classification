import os
import json
import re
import pickle
import random


_DIGIT_RE = re.compile("\d")#数字表达
padId, unkId, eosId, goId = 0, 1, 2, 3

class Batch:
    def __init__(self):
        self.inputs_sentence = []
        self.inputs_sentence_length = []
        self.targets_sentence = []
        self.targets_sentence_length = []
        self.post_cat = []
        self.emo_cat = []

        self.cnn_setences = []
        self.cnn_senteces_length = []
        self.cnn_batch_length = []



def loadDataset(filename):
    dataset_path = os.path.join(filename)
    print('Loading dataset from {}'.format(dataset_path))
    with open(dataset_path,'rb')as handle:
        data = pickle.load(handle)
    return data

def load_test_data(path_s,path_e):
    emotion = []
    sentence = []
    with open(file = path_s,encoding="utf-8") as fs:
        for s in fs:
            sentence.append(s.strip())
    with open(file = path_e,encoding="utf-8") as fe:
        for e in fs:
            emotion.append(e.strip())
    return sentence,emotion

def load_dict():
    dict_file = open('../data/Word_embedding/vocabulary_size_40000', 'r', encoding='utf-8').readlines()
    word2idx = {}
    idx2word = []
    for w in dict_file:
        w = w.strip()
        idx2word.append(w)
        word2idx[w] = len(word2idx)
    # print(word2idx)
    # print(idx2word)
    return word2idx, idx2word

def data2idx(word2idx):
    json_file = open('../data/test_data/new_train_data1.json', 'r')  # '../data/new_test_data1.json'
    data = json.load(json_file)
    data_idx = []
    for pairs in data:
        p_post = []
        for w in pairs[0][0].strip().split(' '):
            word = _DIGIT_RE.sub("0", w)
            if word in word2idx:
                p_post.append(word2idx[word])
            else:
                p_post.append(word2idx['_UNK'])
        post_emo = pairs[0][1]
        p_response = []
        for w in pairs[1][0].strip().split(' '):
            word = _DIGIT_RE.sub("0", w)
            if word in word2idx:
                p_response.append(word2idx[word])
            else:
                p_response.append(word2idx['_UNK'])
        response_emo = pairs[1][1]
        new_pairs = [p_post,post_emo, p_response, response_emo]
        data_idx.append(new_pairs)

    file = open('../data/test1_data_idx.pkl', 'wb')  # '../data/test_data_idx.pkl'
    pickle.dump(data_idx, file)
    # print(len(data))  # 946146
    # print(data[0])
    # print(data_idx[0])

def getBatches(data, batch_size):
    '''
    根据读取出来的所有数据和batch_size将原始数据分成不同的小batch。对每个batch索引的样本调用createBatch函数进行处理
    :param data: loadDataset函数读取之后的trainingSamples
    :param batch_size: batch大小
    :param en_de_seq_len: 列表，第一个元素表示source端序列的最大长度，第二个元素表示target端序列的最大长度
    :return: 列表，每个元素都是一个batch的样本数据，可直接传入feed_dict进行训练
    '''
    # 每个epoch之前都要进行样本的shuffle
    random.shuffle(data)
    batches = []
    data_len = len(data)

    def genNextSamples():
        for i in range(0, data_len, batch_size):
            yield data[i:min(i + batch_size, data_len)]

    for samples in genNextSamples():
        batch = createBatch(samples)
        batches.append(batch)
    return batches


def createBatch(samples):
    '''
    根据给出的samples（就是一个batch的数据），进行padding并构造成placeholder所需要的数据形式
    :param samples: 一个batch的样本数据，列表，每个元素都是[question，emo_cat, answer，emo_cat]的形式，id
    :return: 处理完之后可以直接传入feed_dict的数据格式
    '''
    batch = Batch()
        #sample[0]句子
        #sample[1]标注
        #sample[2]句子
        #sample[3]标注
    for sample in samples:
        print(sample)
    #     batch.inputs_sentence_length.append(len(sample[0]))
    #     batch.targets_sentence_length
    batch.inputs_sentence_length = [len(sample[0]) for sample in samples]
    # print(batch.inputs_sentence_length)
    batch.targets_sentence_length = [len(sample[2]) for sample in samples]
    # print(batch.targets_sentence_length)
    max_source_length = max(batch.inputs_sentence_length)
    max_target_length = max(batch.targets_sentence_length)

    # max_length = max(batch.inputs_sentence_length)
    # batch.cnn_batch_length.append(max_length)
    # print(max_length)


    for sample in samples:
        source = sample[0]
        pad = [padId] * (max_source_length - len(source))
        batch.inputs_sentence.append(source + pad)
        batch.post_cat.append(sample[1])
        target = sample[2]
        pad = [padId] * (max_target_length - len(target))
        batch.targets_sentence.append(target + pad)
        batch.emo_cat.append(sample[3])

    # print('batch.encoder_inputs',batch.encoder_inputs)
    # print('batch.encoder_inputs_length',batch.encoder_inputs_length)
    # print('batch.decoder_targets',batch.decoder_targets)
    # print('batch.decoder_targets_length',batch.decoder_targets_length)
    # print('batch.emo_cat',batch.emo_cat)
    # print(batch)

    # print("batch.inputs_sentence1:", batch.inputs_sentence)
    # print("batch.inputs_sentence1:", len(batch.inputs_sentence))
    # print("batch.emo_cat1_length:", len(batch.emo_cat))
    # print("batch.emo_cat1:",batch.emo_cat)

    return batch
#
if __name__ == '__main__':
    # word2idx, idx2word = load_dict()
    # data2idx(word2idx)
    data_path = '../data/test_data/train_new_data_idx.pkl'
    trainingSamples = loadDataset(data_path)
    #
    batch=getBatches(trainingSamples,1)
    # for e in batch:

        # print(e.inputs_sentence_length)
        # print(len(e.inputs_sentence_length))
        # print(len(e.emo_cat))
        # print(batch)


