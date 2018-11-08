import torch
import torch.nn as nn


class SentenceEncoder(nn.Module):

    def __init__(self, word_dim, embeddings=None, hidden_size=200, num_layers=1, batch_size=40):
        super(SentenceEncoder, self).__init__()
        self.q_hidden_size = hidden_size
        self.s_hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        # self.hidden = self.init_hidden()

        if embeddings is None:
            self.word_embedding = nn.Embedding(91557, word_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)

        self.lstm_q = nn.LSTM(word_dim, self.q_hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm_s = nn.LSTM(word_dim, self.s_hidden_size, num_layers=num_layers, batch_first=True)
        self.last = nn.Linear(self.q_hidden_size + self.s_hidden_size, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, _input, q_hidden, s_hidden):
        # _input = (question, sentence)
        question = _input[0]
        sentence = _input[1]
        q_embedded = self.word_embedding(question)
        s_embedded = self.word_embedding(sentence)

        _, (hq, cn) = self.lstm_q(q_embedded, q_hidden)
        _, (hs, cn) = self.lstm_s(s_embedded, s_hidden)
        hq, hs = hq.squeeze(), hs.squeeze()
        out = self.last(torch.cat((hq, hs), dim=1))
        return self.softmax(out)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.q_hidden_size),
                torch.zeros(self.num_layers, self.batch_size, self.q_hidden_size))
