import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import ujson as json
from const import *
from model import NLIEncoder
from tqdm import tqdm


def eval(model, criterion, valid_data):
    total_loss = 0
    total_num_correct = 0
    random.shuffle(valid_data)

    yes_answer = 0
    no_answer = 0
    yes_answer_correct = 0
    no_answer_correct = 0

    for i in tqdm(range(len(valid_data))):
        _input = valid_data[i][0]
        _target = valid_data[i][1]

        batch_input = [_input[0].unsqueeze(0), _input[1].unsqueeze(0)]

        # get last layer's output(using [-1])
        output = model.transfer(batch_input)
        loss = criterion(output, _target.unsqueeze(0))
        total_loss += float(loss)

        _, i = torch.max(output[0], 0)

        if _target.item() == 0:
            yes_answer += 1
        else:
            no_answer += 1

        if i.item() == _target.item():
            total_num_correct += 1
            if i.item() == 0:
                yes_answer_correct += 1
            else:
                no_answer_correct += 1

    print("YES ANSWER: %.2f (%d/%d), NO ANSWER: %.2f (%d/%d)" % (
        yes_answer_correct/yes_answer*100,
        yes_answer_correct, yes_answer,
        no_answer_correct/no_answer*100,
        no_answer_correct, no_answer))
    return total_num_correct/len(valid_data)*100, total_loss/len(valid_data)


def transfer(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = torch.tensor(json.load(fh), device=device)
    with open(config.transfer_train_file, "r") as fh:
        raw_train_data = json.load(fh)
        train_data = [[
            (torch.tensor(d["question_idx"], device=device),
             torch.tensor(d["context_idx"], device=device)),
            torch.tensor(0, device=device) if d["is_impossible"] is False else torch.tensor(1, device=device)] for d in raw_train_data]
    with open(config.transfer_dev_file, "r") as fh:
        raw_dev_data = json.load(fh)
        dev_data = [[
            (torch.tensor(d["question_idx"], device=device),
             torch.tensor(d["context_idx"], device=device)),
            torch.tensor(0, device=device) if d["is_impossible"] is False else torch.tensor(1, device=device)] for d in raw_dev_data]

    train_loader = data.DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True)

    model = NLIEncoder(config.glove_dim, embeddings=word_mat, batch_size=config.batch_size).to(device)
    model.load_state_dict(torch.load(config.snli_test_model))

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
            model.zero_grad()
            tag_scores = model.transfer(batch_input)
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

        # eval
        valid_acc, valid_loss = eval(model, criterion, dev_data)
        print("[%d/%d]: Loss %.3f, Accuracy: %.3f" % (epoch+1, config.epoch, valid_loss, valid_acc))
