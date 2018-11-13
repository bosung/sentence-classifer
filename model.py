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

        self.lstm = nn.LSTM(word_dim, self.q_hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True)
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

        _, (hq, cn) = self.lstm(q_embedded)
        _, (hs, cn) = self.lstm(s_embedded)
        hq1, hq2, hs1, hs2 = hq[0], hq[1], hs[0], hs[1]
        sub = hq1 + (-hs1)
        mul = torch.mul(hq1, hs1)
        out = self.last(torch.cat((torch.cat((hq1, hq2), dim=1),
                                   torch.cat((sub, mul), dim=1)), dim=1))
        # return self.softmax(out)  # Logsoftmax with NLLLoss
        return self.sigmoid(out)  # sigmoid with BCELoss

    def sent_embed(self, _input):
        embedded = self.word_embedding(_input)

        _, (hq, cn) = self.lstm(embedded)
        return hq

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return (torch.zeros(self.num_layers, batch_size, self.q_hidden_size),
                torch.zeros(self.num_layers, batch_size, self.q_hidden_size))
