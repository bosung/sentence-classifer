import ujson as json
from tqdm import tqdm
from model import SentenceEncoder
from const import *
import torch
import torch.nn as nn

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


def eval(config):
    """ sentence similarity evaluation """
    with open(config.sim_eval_test, "r") as fh:
        test_data = json.load(fh)
    with open(config.word_emb_file, "r") as fh:
        word_mat = torch.tensor(json.load(fh), device=device)

    accuracy1 = 0
    accuracy2 = 0
    accuracy3 = 0

    model = SentenceEncoder(config.glove_dim, embeddings=word_mat, batch_size=config.batch_size).to(device)
    # model.load_state_dict(torch.load(config.test_model))

    print("average-of-word-embedding")
    for i in tqdm(range(len(test_data))):
        answer_idx = list(set(test_data[i]["answer_idx"]))
        _input = torch.tensor(test_data[i]["question_idx"]).to(device)
        _target = torch.tensor(test_data[i]["context_sent_list_idx"]).to(device)

        q = model.sent_embed(_input.unsqueeze(0))
        s = model.sent_embed(_target)

        temp = {}
        for j in range(s.size(0)):
            temp[j] = cos(q, s[j].unsqueeze(0))

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

    print("model: %s, accuracy@1: %.3f, accuracy@2: %.3f, accuracy@3: %.3f" % (
        config.test_model,
        (accuracy1/len(test_data)*100),
        (accuracy2/len(test_data)*100),
        (accuracy3/len(test_data)*100)))
