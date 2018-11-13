import ujson as json
from tqdm import tqdm
from model import SentenceEncoder
from const import *
import torch
import torch.nn as nn

cos = nn.CosineSimilarity()


def eval(config):
    """ sentence similarity evaluation """
    with open(config.test_file, "r") as fh:
        test_data = json.load(fh)
        #test_data = [[
        #    torch.tensor(d["question_idx"], device=device),
        #    torch.tensor(d["context_sents"], device=device)] for d in raw_test_data]

    total_num_correct = 0

    model = SentenceEncoder(config.glove_dim, batch_size=config.batch_size).to(device)
    # model.load_state_dict(torch.load(config.test_model))

    for i in tqdm(range(len(test_data))):
        _input = torch.tensor(test_data[i]["question"]).to(device)
        _target = torch.tensor(test_data[i]["context_sent_list"]).to(device)
        # for batch

        # get last layer's output(using [-1]) and only one result(using [0]) from batch
        q_f, q_b = model.sent_embed(_input)
        q = torch.cat((q_f, q_b), dim=1)
        s_f, s_b = model.sent_embed(_target)
        s = torch.cat((s_f, s_b), dim=1)

        temp = []
        for j in range(s.size(0)):
            print(j)
            temp.append(cos(q, s[j]))

    print("model: %s, accuracy: %.3f" % (config.test_model, (total_num_correct/len(test_data)*100)))
