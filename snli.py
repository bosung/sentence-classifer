import random
import ujson as json
import torch.nn as nn
import torch.utils.data as data
from torch import optim
from tqdm import tqdm
from model import SentenceEncoder
from const import *


def eval(model, criterion, valid_data):
    total_loss = 0
    total_num_correct = 0
    random.shuffle(valid_data)

    for i in tqdm(range(len(valid_data))):
        _input = valid_data[i][0]
        _target = valid_data[i][1]

        batch_input = _input

        # get last layer's output(using [-1])
        output = model(batch_input)
        loss = criterion(output, _target.unsqueeze(0))
        total_loss += float(loss)

        _, i = torch.max(output[0], 0)
        if i.item() == _target.item():
            total_num_correct += 1

    return total_num_correct/len(valid_data)*100, total_loss/len(valid_data)


def train(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = torch.tensor(json.load(fh), device=device)
    with open(config.snli_train_file, "r") as fh:
        raw_train_data = json.load(fh)
        train_data = [[
            ([torch.tensor(d["premise_idx"]).to(device), torch.tensor(d["premise_char_idx"]).to(device)],
             [torch.tensor(d["hypothesis_idx"]).to(device), torch.tensor(d["hypothesis_char_idx"]).to(device)]),
            torch.tensor(d["label_idx"], device=device)] for d in raw_train_data]
    with open(config.snli_dev_file, "r") as fh:
        raw_dev_data = json.load(fh)
        dev_data = [[
            ([torch.tensor(d["premise_idx"]).to(device), torch.tensor(d["premise_char_idx"]).to(device)],
             [torch.tensor(d["hypothesis_idx"]).to(device), torch.tensor(d["hypothesis_char_idx"]).to(device)]),
            torch.tensor(d["label_idx"], device=device)] for d in raw_dev_data]

    train_loader = data.DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True)

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

    for epoch in range(config.epoch):
        for i, (batch_input, batch_target) in enumerate(train_loader):
            model.zero_grad()  # torch accumulate gradients, making them zero for each minibatch
            tag_scores = model(batch_input)
            # tag_scores = (num_layers, batch_size, class_num)
            # maybe last layer's output might be tag_score[-1]
            loss = criterion(tag_scores, batch_target.squeeze())
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

        # eval
        valid_acc, valid_loss = eval(model, criterion, dev_data)
        print("[%d/%d]: Loss %.3f, Accuracy: %.3f" % (epoch+1, config.epoch, valid_loss, valid_acc))

        # save checkpoint
        if valid_acc > best_accuracy:
            checkpoint = {
                'model': model,
                'optim': optimizer,
                'loss': valid_loss
            }

            # save model state dict
            print("New record [%d/%d]: %.3f" % (epoch+1, config.epoch, valid_acc))
            print(checkpoint)
            torch.save(model.state_dict(), config.snli_state_dict + str(epoch) + '-' + str(round(valid_acc, 2)))
            best_accuracy = valid_acc
        else:
            # adjust lr
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.85


def test(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = torch.tensor(json.load(fh), device=device)
    with open(config.snli_test_file, "r") as fh:
        raw_test_data = json.load(fh)
        test_data = [[
            (torch.tensor(d["premise_idx"], device=device),
             torch.tensor(d["hypothesis_idx"], device=device)),
            torch.tensor(d["label_idx"], device=device)] for d in raw_test_data]

    total_num_correct = 0

    model = SentenceEncoder(config.glove_dim, embeddings=word_mat, batch_size=config.batch_size).to(device)
    model.load_state_dict(torch.load(config.snli_test_model))

    for i in tqdm(range(len(test_data))):
        _input = test_data[i][0]
        _target = test_data[i][1]

        batch_input = [_input[0].unsqueeze(0), _input[1].unsqueeze(0)]

        # get last layer's output(using [-1]) and only one result(using [0]) from batch
        output = model(batch_input)[0]

        _, i = torch.max(output, 0)
        if i.item() == _target.item():
            total_num_correct += 1

    print("model: %s, accuracy: %.3f" % (config.snli_test_model, (total_num_correct/len(test_data)*100)))
