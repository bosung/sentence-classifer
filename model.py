import torch
import torch.nn as nn


class SentenceEncoder(nn.Module):

    def __init__(self, word_dim, embeddings=None, hidden_size=200, num_layers=1, batch_size=40):
        super(SentenceEncoder, self).__init__()
        self.q_hidden_size = hidden_size
        self.s_hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        if embeddings is None:
            self.word_embedding = nn.Embedding(91557, word_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)

        self.lstm = nn.LSTMCell(word_dim, self.q_hidden_size)
        self.last = nn.Linear(self.q_hidden_size * 4, 2)
        self.softmax = nn.LogSoftmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, _input):
        # _input = (question, sentence)
        question = _input[0]
        sentence = _input[1]
        q_embedded = self.word_embedding(question)
        s_embedded = self.word_embedding(sentence)

        # max pooling
        # hq = self.max_pooling(q_embedded)
        # hs = self.max_pooling(s_embedded)

        # self_attn + max pooling
        hq, hs = self.self_attn(q_embedded, s_embedded)

        sub = hq + (-hs)
        mul = torch.mul(hq, hs)
        out = self.last(torch.cat((torch.cat((hq, hs), dim=1),
                                   torch.cat((sub, mul), dim=1)), dim=1))
        # return self.softmax(out)  # Logsoftmax with NLLLoss

        # out = self.last(torch.cat((hq, hs), dim=1))
        return self.sigmoid(out)  # sigmoid with BCELoss

    def get_prob(self, q, s):
        sub = q + (-s)
        mul = torch.mul(q, s)
        out = self.last(torch.cat((torch.cat((q, s), dim=1),
                                   torch.cat((sub, mul), dim=1)), dim=1))
        # return self.last(torch.cat((q, s), dim=1))[0][1]
        return out[0][1]

    def self_attn(self, _input, _target):
        _i = self.get_hidden_matrix(_input)
        _t = self.get_hidden_matrix(_target)
        _tt = _t.transpose(1, 2)

        attn_mat = torch.bmm(_i, _tt)
        attn_mat1 = self.softmax_2(attn_mat)
        output1 = torch.bmm(attn_mat1, _t)
        hq = torch.max(output1, dim=1)[0]

        attn_mat2 = self.softmax_2(attn_mat.transpose(1, 2))
        output2 = torch.bmm(attn_mat2, _i)
        hs = torch.max(output2, dim=1)[0]

        return hq, hs

    def get_hidden_matrix(self, _input):
        vector = _input.transpose(1, 0)
        # vector -> (max_length, batch_size, input_size)

        hx, cx = self.lstm(vector[0])
        output = hx.unsqueeze(1)
        # hx -> (batch, 1, hidden_size)
        for i in range(1, vector.size(0)):
            hx, cx = self.lstm(vector[i], (hx, cx))
            output = torch.cat((output, hx.unsqueeze(1)), dim=1)

        return output

    def max_pooling(self, _input):
        output = self.get_hidden_matrix(_input)
        return torch.max(output, dim=1)[0]

    def sent_embed(self, _input):
        embedded = self.word_embedding(_input)
        """
        # 1. encoder last hidden
        _, (hq, cn) = self.lstm(embedded)
        _f, _b = hq
        return torch.cat((_f, _b), dim=1)

        # 2. word embedding avg
        hq = torch.mean(embedded, dim=1).squeeze(1)
        return hq
        """
        # 3. max pooling
        return self.max_pooling(embedded)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return (torch.zeros(self.num_layers, batch_size, self.q_hidden_size),
                torch.zeros(self.num_layers, batch_size, self.q_hidden_size))


class NLIEncoder(nn.Module):

    def __init__(self, word_dim, embeddings=None, hidden_size=200, num_layers=3, batch_size=40):
        super(NLIEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        # self.hidden = self.init_hidden()

        if embeddings is None:
            self.word_embedding = nn.Embedding(36906, word_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)

        self.lstm = nn.LSTM(word_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm_cell = nn.LSTMCell(word_dim, self.hidden_size)
        self.last = nn.Linear(hidden_size * 4, 3)
        self.softmax = nn.LogSoftmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)
        self.bi_clf = nn.Linear(hidden_size * 4, 2)

    def self_attn(self, _input, _target):
        _i = self.get_hidden_matrix(self.word_embedding(_input))
        _t = self.get_hidden_matrix(self.word_embedding(_target))
        _tt = _t.transpose(1, 2)

        attn_mat = torch.bmm(_i, _tt)
        attn_mat1 = self.softmax_2(attn_mat)
        output1 = torch.bmm(attn_mat1, _t)
        hq = torch.max(output1, dim=1)[0]

        attn_mat2 = self.softmax_2(attn_mat.transpose(1, 2))
        output2 = torch.bmm(attn_mat2, _i)
        hs = torch.max(output2, dim=1)[0]

        return hq, hs

    def get_hidden_matrix(self, _input):
        vector = _input.transpose(1, 0)
        # vector -> (max_length, batch_size, input_size)

        hx, cx = self.lstm_cell(vector[0])
        output = hx.unsqueeze(1)
        # hx -> (batch, 1, hidden_size)
        for i in range(1, vector.size(0)):
            hx, cx = self.lstm_cell(vector[i], (hx, cx))
            output = torch.cat((output, hx.unsqueeze(1)), dim=1)

        return output

    def max_pooling(self, _input):
        output = self.get_hidden_matrix(_input)
        return torch.max(output, dim=1)[0]

    def get_sent_embed(self, _input):
        x = self.word_embedding(_input)
        """
        # 1. LSTM last hidden
        _, (h, c) = self.lstm(x)
        output = h[-1]
        """
        # 2. LSTM max pooling
        output = self.max_pooling(x)
        return output

    def forward(self, _input):
        # input: [premise, hypothesis]
        premise = _input[0]
        hypothesis = _input[1]

        # hp = self.get_sent_embed(premise)
        # hh = self.get_sent_embed(hypothesis)
        hp, hh = self.self_attn(premise, hypothesis)

        sub = hp + (-hh)
        mul = torch.mul(hp, hh)
        out = self.last(torch.cat((torch.cat((hp, hh), dim=1),
                                   torch.cat((sub, mul), dim=1)), dim=1))
        return self.softmax(out)

    def transfer(self, _input):
        question = _input[0]
        context = _input[1]

        hp = self.get_sent_embed(question)
        hh = self.get_sent_embed(context)

        sub = hp + (-hh)
        mul = torch.mul(hp, hh)
        out = self.bi_clf(torch.cat((torch.cat((hp, hh), dim=1),
                                     torch.cat((sub, mul), dim=1)), dim=1))
        return self.softmax(out)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
