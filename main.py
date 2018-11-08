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


def eval(model, criterion, valid_data, batch_size):
    total_loss = 0
    total_num_correct = 0

    for i in tqdm(range(len(valid_data))):
        _input = valid_data[i][0]
        _target = valid_data[i][1]
        # for batch
        batch_input = _input.unsqueeze(0)
        for _ in range(batch_size - 1):
            batch_input = torch.cat((batch_input, _input.unsqueeze(0)), dim=0)

        hq0, cq0 = model.init_hidden()
        hq0, cq0 = hq0.to(device), cq0.to(device)

        hs0, cs0 = model.init_hidden()
        hs0, cs0 = hs0.to(device), cs0.to(device)

        # get last layer's output(using [-1]) and only one result(using [0]) from batch
        output = model(batch_input, (hq0, cq0), (hs0, cs0))[-1][0]
        loss = criterion(output.unsqueeze(0), _target.unsqueeze(0))
        total_loss += float(loss)

        _, i = torch.max(output, 0)
        if i.item() == _target.item():
            total_num_correct += 1

    return total_num_correct/len(valid_data)*100, total_loss/len(valid_data)


def train(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = torch.tensor(json.load(fh), device=device)
    with open(config.train_file, "r") as fh:
        raw_train_data = json.load(fh)
        train_data = [[
            (torch.tensor(d["question_idx"], device=device),
             torch.tensor(d["sentence_idx"], device=device)),
            torch.tensor(d["label_idx"], device=device)] for d in raw_train_data]
    with open(config.dev_file, "r") as fh:
        raw_dev_data = json.load(fh)
        dev_data = [[
            (torch.tensor(d["question_idx"], device=device),
             torch.tensor(d["sentence_idx"], device=device)),
            torch.tensor(d["label_idx"], device=device)] for d in raw_dev_data]

    train_loader = data.DataLoader(dataset=train_data, batch_size=config.batch_size)

    model = SentenceEncoder(config.glove_dim, embeddings=word_mat, batch_size=config.batch_size).to(device)

    if config.optim == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)
    elif config.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

    criterion = nn.NLLLoss().to(device)

    best_accuracy = 0
    local_loss = 0

    hq0, cq0 = model.init_hidden()
    hq0, cq0 = hq0.to(device), cq0.to(device)

    hs0, cs0 = model.init_hidden()
    hs0, cs0 = hs0.to(device), cs0.to(device)

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
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (
                    epoch + 1,
                    config.epoch,
                    i + 1, len(train_data) // config.batch_size,
                    local_loss / 400))
                local_loss = 0
        """
        # eval
        valid_acc, valid_loss = eval(model, criterion, dev_data, config.batch_size)
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
        """


def test(config):
    with open(config.test_file, "r") as fh:
        raw_test_data = json.load(fh)
        test_data = [[
            torch.tensor(d["premise_idx"]+d["hypothesis_idx"], device=device),
            torch.tensor(d["label_idx"], device=device)] for d in raw_test_data]

    total_num_correct = 0

    model = Encoder(config.glove_dim, batch_size=config.batch_size).to(device)
    model.load_state_dict(torch.load(config.test_model))

    for i in tqdm(range(len(test_data))):
        _input = test_data[i][0]
        _target = test_data[i][1]
        # for batch
        batch_input = _input.unsqueeze(0)
        for _ in range(config.batch_size - 1):
            batch_input = torch.cat((batch_input, _input.unsqueeze(0)), dim=0)

        h0, c0 = model.init_hidden()
        h0, c0 = h0.to(device), c0.to(device)

        # get last layer's output(using [-1]) and only one result(using [0]) from batch
        output = model(batch_input, h0, c0)[-1][0]

        _, i = torch.max(output, 0)
        if i.item() == _target.item():
            total_num_correct += 1

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
    else:
        print("Unknown mode")
        exit(0)
