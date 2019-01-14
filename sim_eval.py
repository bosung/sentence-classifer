import ujson as json
from tqdm import tqdm
from model import SentenceEncoder
from const import *
import torch
import torch.nn as nn
from prepro import get_word, char2idx

cos = nn.CosineSimilarity()


def get_top_n(data, n):
    sorted_dict = {}
    num = 0
    for key, value in reversed(sorted(data.items(), key=lambda i: (i[1], i[0]))):
        sorted_dict[key] = value
        num += 1
        if num == n:
            break
    return sorted_dict


def word2char_idx(tokens, _max_length):
    result = list()
    for t in tokens:
        result.append(char2idx(t, c_max_length=_max_length))
    return result


def convert2idx(tokens, w2i_dict):
    result = list()
    for _token in tokens:
        result.append(get_word(w2i_dict, _token))
    return result


def eval(config):
    """ sentence similarity evaluation """
    with open(config.sim_eval_test, "r") as fh:
        test_data = json.load(fh)
    with open(config.word_emb_file, "r") as fh:
        word_mat = torch.tensor(json.load(fh)).to(device)
    with open(config.word2idx_file, "r") as fh:
        word2idx = json.load(fh)

    accuracy1 = 0
    accuracy2 = 0
    accuracy3 = 0

    mrr = 0

    model = SentenceEncoder(config.glove_dim, embeddings=word_mat, batch_size=config.batch_size).to(device)
    model.load_state_dict(torch.load(config.test_model))

    for i in tqdm(range(len(test_data))):
        doc = test_data[i]
        answer_idx = list(set(doc["answer_idx"]))

        _input = [torch.tensor(convert2idx(doc["q_tokens"], word2idx)).to(device),
                  torch.tensor(word2char_idx(doc["q_tokens"], _max_length=8)).to(device)]
        # q = model.get_sent_embed(_input)

        temp = {}
        for j, sent in enumerate(doc["context_sent_list"]):
            candi = sent[2]  # sent[2]: tokens
            _target = [torch.tensor(convert2idx(candi, word2idx)).to(device),
                       torch.tensor(word2char_idx(candi, _max_length=8)).to(device)]
            # s = model.get_sent_embed(_target)
            q, s = model.dense_attn_rnn(_input, _target)
            temp[j] = model.get_prob(q, s)
            # temp[j] = cos(q, s)

        """
        # self attention
        i_e = model.word_embedding(_input.unsqueeze(0))
        t_e = model.word_embedding(_target)

        temp = {}
        for j in range(t_e.size(0)):
            q, s = model.self_attn(i_e, t_e[j].unsqueeze(0))
            temp[j] = model.get_prob(q, s)
        
        # calc accuracy
        sorted_score = get_top_n(temp, 3)
        for k in sorted_score:
            if k in answer_idx:
                accuracy3 += 1
                break

        sorted_score = get_top_n(temp, 2)
        for k in sorted_score:
            if k in answer_idx:
                accuracy2 += 1
                break

        sorted_score = get_top_n(temp, 1)
        for k in sorted_score:
            if k in answer_idx:
                accuracy1 += 1
                break
        """
        sorted_score = get_top_n(temp, len(temp))
        for idx, k in enumerate(sorted_score, 1):
            if k in answer_idx:
                mrr += (1/idx)

                if idx == 1:
                    accuracy1 += 1
                    accuracy2 += 1
                    accuracy3 += 1
                elif idx == 2:
                    accuracy2 += 1
                    accuracy3 += 1
                elif idx == 3:
                    accuracy3 += 1

    print("model:%s, accuracy@1: %.3f, accuracy@2: %.3f, accuracy@3: %.3f, MRR: %.2f" % (
        config.test_model,
        (accuracy1/len(test_data)*100),
        (accuracy2/len(test_data)*100),
        (accuracy3/len(test_data)*100),
        (mrr/len(test_data)*100)))
