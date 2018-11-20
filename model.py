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
        self.sigmoid = nn.Sigmoid()

    # def forward(self, _input, q_hidden, s_hidden):
    def forward(self, _input):
        # _input = (question, sentence)
        question = _input[0]
        sentence = _input[1]
        q_embedded = self.word_embedding(question)
        s_embedded = self.word_embedding(sentence)

        # max pooling
        hq = self.max_pooling(q_embedded)
        hs = self.max_pooling(s_embedded)

        # sub = hq + (-hs)
        # mul = torch.mul(hq, hs)
        # out = self.last(torch.cat((torch.cat((hq, hs), dim=1),
        #                            torch.cat((sub, mul), dim=1)), dim=1))
        # return self.softmax(out)  # Logsoftmax with NLLLoss

        out = self.last(torch.cat((hq, hs), dim=1))
        return self.sigmoid(out)  # sigmoid with BCELoss

    def max_pooling(self, _input):
        vector = _input.transpose(1, 0)
        # vector -> (max_length, batch_size, input_size)

        hx, cx = self.lstm(vector[0])
        output = hx.unsqueeze(1)
        # hx -> (batch, 1, hidden_size)
        for i in range(1, vector.size(0)):
            hx, cx = self.lstm(vector[i], (hx, cx))
            output = torch.cat((output, hx.unsqueeze(1)), dim=1)

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
