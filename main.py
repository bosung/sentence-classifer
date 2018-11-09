import argparse
import prepro
import ujson as json
import torch.nn as nn
import torch.utils.data as data
from torch import optim
from tqdm import tqdm
from config import Config
from model import SentenceEncoder
from const import *


def eval(model, criterion, valid_data):
    total_loss = 0
    total_num_correct = 0
    # gold sentence
    total_gold_sent = 0
    gold_sent_correct = 0
    # irrelevant sentence
    total_ir_sent = 0
    ir_correct = 0

    for i in tqdm(range(len(valid_data))):
        _input = valid_data[i][0]
        _target = valid_data[i][1]
        _, ti = torch.max(_target, 0)
        if ti.item() == 1:
            total_gold_sent += 1
        else:
            total_ir_sent += 1

        hq0, cq0 = model.init_hidden(batch_size=1)
        hq0, cq0 = hq0.to(device), cq0.to(device)

        hs0, cs0 = model.init_hidden(batch_size=1)
        hs0, cs0 = hs0.to(device), cs0.to(device)

        _input = [_input[0].unsqueeze(0), _input[1].unsqueeze(0)]
        output = model(_input, (hq0, cq0), (hs0, cs0))
        loss = criterion(output, _target.unsqueeze(0))
        total_loss += float(loss)

        _, i = torch.max(output[0], 0)
        _, ti = torch.max(_target, 0)
        if i.item() == ti.item():
            total_num_correct += 1
            if ti.item() == 1:
                gold_sent_correct += 1
            else:
                ir_correct += 1

    print("gold sent accuracy: %.3f / ir sent accuracy: %.3f" % (
            gold_sent_correct/total_gold_sent*100,
            ir_correct/total_ir_sent*100))
    return total_num_correct/len(valid_data)*100, total_loss/len(valid_data)


def train(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = torch.tensor(json.load(fh), device=device)
    with open(config.train_file, "r") as fh:
        raw_train_data = json.load(fh)
        train_data = [[
            (torch.tensor(d["question_idx"], device=device),
             torch.tensor(d["sentence_idx"], device=device)),
            torch.tensor([1., 0.], device=device) if d["label_idx"] == 0 else torch.tensor([0., 1.], device=device)] for d in raw_train_data]
    with open(config.dev_file, "r") as fh:
        raw_dev_data = json.load(fh)
        dev_data = [[
            (torch.tensor(d["question_idx"], device=device),
             torch.tensor(d["sentence_idx"], device=device)),
            torch.tensor([1., 0.], device=device) if d["label_idx"] == 0 else torch.tensor([0., 1.], device=device)] for d in raw_dev_data]

    train_loader = data.DataLoader(dataset=train_data, batch_size=config.batch_size)

    model = SentenceEncoder(config.glove_dim, embeddings=word_mat, batch_size=config.batch_size).to(device)

    if config.optim == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)
    elif config.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

    criterion = nn.BCELoss().to(device)

    best_accuracy = 0
    local_loss = 0

    hq0, cq0 = model.init_hidden()
    hq0, cq0 = hq0.to(device), cq0.to(device)

    hs0, cs0 = model.init_hidden()
    hs0, cs0 = hs0.to(device), cs0.to(device)

    print("training...")
    for epoch in range(config.epoch):
        for i, (batch_input, batch_target) in enumerate(train_loader):
            model.zero_grad()  # torch accumulate gradients, making them zero for each minibatch
            tag_scores = model(batch_input, (hq0, cq0), (hs0, cs0))
            # tag_scores = (num_layers, batch_size, class_num)
            loss = criterion(tag_scores, batch_target)
            loss.backward()
            optimizer.step()
            local_loss += float(loss)

            if (i + 1) % 400 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.8f' % (
                    epoch + 1,
                    config.epoch,
                    i + 1, len(train_data) // config.batch_size,
                    local_loss / 400))
                local_loss = 0

        # eval
        valid_acc, valid_loss = eval(model, criterion, dev_data)
        print("[%d/%d]: Loss %.3f, Accuracy: %.3f" % (epoch+1, config.epoch, valid_loss, valid_acc))

        # save checkpoint
        if (epoch + 1) % config.checkpoint == 0 or (epoch + 1) == config.epoch:
            checkpoint = {
                'model': model,
                'optim': optimizer,
                'loss': valid_loss
            }
            if valid_acc > best_accuracy:
                # save model state dict
                print("New record [%d/%d]: %.3f" % (epoch+1, config.epoch, valid_acc))
                print(checkpoint)
                torch.save(model.state_dict(), config.state_dict+str(epoch))
                best_accuracy = valid_acc


def test(config):
    with open(config.test_file, "r") as fh:
        raw_test_data = json.load(fh)
        test_data = [[
            (torch.tensor(d["question_idx"], device=device),
             torch.tensor(d["sentence_idx"], device=device)),
            torch.tensor(d["label_idx"], device=device)] for d in raw_test_data]

    total_num_correct = 0
    # gold sentence
    total_gold_sent = 0
    gold_sent_correct = 0
    # irrelevant sentence
    total_ir_sent = 0
    ir_correct = 0

    model = SentenceEncoder(config.glove_dim, batch_size=config.batch_size).to(device)
    model.load_state_dict(torch.load(config.test_model))

    for i in tqdm(range(len(test_data))):
        _input = test_data[i][0]
        _target = test_data[i][1]
        if _target.item() == 1:
            total_gold_sent += 1
        else:
            total_ir_sent += 1

        hq0, cq0 = model.init_hidden(batch_size=1)
        hq0, cq0 = hq0.to(device), cq0.to(device)

        hs0, cs0 = model.init_hidden(batch_size=1)
        hs0, cs0 = hs0.to(device), cs0.to(device)

        # get last layer's output(using [-1]) and only one result(using [0]) from batch
        output = model(_input, (hq0, cq0), (hs0, cs0))[0]

        _, i = torch.max(output, 0)
        if i.item() == _target.item():
            total_num_correct += 1
            if _target.item() == 1:
                gold_sent_correct += 1
            else:
                ir_correct += 1

    print("model: %s, accuracy: %.3f" % (config.test_model, (total_num_correct/len(test_data)*100)))
    print("gold sent accuracy: %.3f / ir sent accuracy: %.3f" % (
        gold_sent_correct/total_gold_sent*100,
        ir_correct/total_ir_sent*100))


def sim_eval(config):
    """ sentence similarity evaluation """
    with open(config.test_file, "r") as fh:
        raw_test_data = json.load(fh)
        test_data = [[
            torch.tensor(d["premise_idx"]+d["hypothesis_idx"], device=device),
            torch.tensor(d["label_idx"], device=device)] for d in raw_test_data]

    total_num_correct = 0

    model = SentenceEncoder(config.glove_dim, batch_size=config.batch_size).to(device)
    model.load_state_dict(torch.load(config.test_model))

    for i in tqdm(range(len(test_data))):
        _input = test_data[i][0]
        _target = test_data[i][1]
        # for batch
        batch_input = _input.unsqueeze(0)
        for _ in range(config.batch_size - 1):
            batch_input = torch.cat((batch_input, _input.unsqueeze(0)), dim=0)

        hq0, cq0 = model.init_hidden()
        hq0, cq0 = hq0.to(device), cq0.to(device)

        hs0, cs0 = model.init_hidden()
        hs0, cs0 = hs0.to(device), cs0.to(device)

        # get last layer's output(using [-1]) and only one result(using [0]) from batch
        q, s = model.sent_embed(batch_input, (hq0, cq0), (hs0, cs0))[0]

    print("model: %s, accuracy: %.3f" % (config.test_model, (total_num_correct/len(test_data)*100)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='debug')
    args = parser.parse_args()

    config = Config()

    if args.mode == 'prepro':
        prepro.prepro(config)
    elif args.mode == 'train':
        train(config)
    elif args.mode == 'debug':
        config.epoch = 1
        train(config)
    elif args.mode == 'test':
        test(config)
    elif args.mode == 'sim-eval':
        sim_eval(config)
    else:
        print("Unknown mode")
        exit(0)
