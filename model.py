import torch
import torch.nn as nn
import torch.nn.functional as F


class SentenceEncoder(nn.Module):

    def __init__(self, word_dim, embeddings=None, hidden_size=100, num_layers=1, batch_size=40):
        super(SentenceEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.char_dim = 16
        self.kernel_size = 3
        self.em = 3

        if embeddings is None:
            self.word_embedding = nn.Embedding(91557, word_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)

        # char embedding
        self.char_embedding = nn.Embedding(53, self.char_dim)
        self.char_lstm = nn.LSTM(self.char_dim, self.char_dim, batch_first=True)

        # nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0 ...)
        # in_channels -> 1
        # out_channels -> char_dim (the number of filters)
        self.conv1d = nn.Conv1d(1, self.char_dim, self.kernel_size)

        self.lstm = nn.LSTM((word_dim + self.char_dim),
                            hidden_size,
                            num_layers=num_layers,
                            bidirectional=True)
        self.lstm_cell = nn.LSTMCell((word_dim + self.char_dim), self.hidden_size)
        self.lstm_cell_2 = nn.LSTMCell((word_dim + self.char_dim) + (self.hidden_size * 2), self.hidden_size)
        self.lstm_cell_3 = nn.LSTMCell((word_dim + self.char_dim) + (self.hidden_size * 4), self.hidden_size)
        self.lstm_cell_4 = nn.LSTMCell((word_dim + self.char_dim) + (self.hidden_size * 6), self.hidden_size)
        self.lstm_cell_5 = nn.LSTMCell((word_dim + self.char_dim) + (self.hidden_size * 8), self.hidden_size)

        self.fcl = nn.Linear(self.hidden_size * 5, 800)
        self.last = nn.Linear(800, 2)
        self.softmax = nn.LogSoftmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        # drop out
        self.drop_out2 = nn.Dropout(p=0.2)
        self.drop_out5 = nn.Dropout(p=0.5)

    def forward(self, _input):
        # _input = (question, sentence)
        question = _input[0]
        sentence = _input[1]

        # hq = self.get_sent_embed(question)
        # hs = self.get_sent_embed(sentence)
        hq, hs = self.dense_attn_rnn(question, sentence)

        sub = hs + (-hq)
        mul = torch.mul(hq, hs)
        _abs = torch.abs(hq-hs)
        out = self.last(F.relu(self.fcl(torch.cat([hq, hs, sub, mul, _abs], dim=1))))
        # return self.softmax(out)  # Logsoftmax with NLLLoss
        return self.sigmoid(out)  # sigmoid with BCELoss

    def get_prob(self, hq, hs):
        sub = hs + (-hq)
        mul = torch.mul(hq, hs)
        _abs = torch.abs(hq-hs)
        out = self.last(F.relu(self.fcl(torch.cat([hq, hs, sub, mul, _abs], dim=1))))
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
        vector = _input
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

    def get_embed(self, _input):
        words = _input[0]
        chars = _input[1]

        if words.dim() < 2:
            words = words.unsqueeze(0)
            chars = chars.unsqueeze(0)

        w_e = self.word_embedding(words)
        # w_e_fix = self.word_embedding_fix(words)

        # CNN char embedding
        # chars = (batch_size=64, max_len=15, max_char_len=7)
        # char_embed -> (batch_size=64, max_len=15, char_dim=16)
        cnn_input = self.char_embedding(chars[0]).view(chars.size(1), 1, -1)
        char_embed, _ = self.conv1d(cnn_input).max(dim=2)
        char_embed = char_embed.unsqueeze(0)
        for i in range(1, chars.size(0)):
            # let max_len(chars.size(1)) be batch size of CNN
            cnn_input = self.char_embedding(chars[i]).view(chars.size(1), 1, -1)
            _temp, _ = F.relu(self.conv1d(cnn_input)).max(dim=2)
            char_embed = torch.cat((char_embed, _temp.unsqueeze(0)), dim=0)

        """
        # LSTM char embedding
        # char embedding -> (batch_size, max_len, char_embed_size)
        _, (char_embed, _) = self.char_lstm(self.char_embedding(chars[0]))
        for i in range(1, chars.size(0)):
            # _temp -> (batch_size=30, seq_len=7, input_size=15)
            _temp = self.char_embedding(chars[i])
            _, (hn, cn) = self.char_lstm(_temp)
            char_embed = torch.cat((char_embed, hn), dim=0)
        """

        # return self.drop_out5(torch.cat([w_e, w_e_fix, char_embed], dim=2).transpose(1, 0))
        return torch.cat([w_e, char_embed], dim=2).transpose(1, 0)

    def co_attn(self, hps, hhs):
        hhs_t = hhs.transpose(1, 2)

        attn_mat = torch.bmm(hps, hhs_t)
        attn_mat_prob = self.softmax_2(attn_mat)
        a_p_mat = torch.bmm(attn_mat_prob, hhs)

        attn_mat_prob2 = self.softmax_2(attn_mat.transpose(1, 2))
        a_h_mat = torch.bmm(attn_mat_prob2, hps)

        return a_p_mat, a_h_mat

    def dense_attn_rnn(self, _input, _target):
        models = [self.lstm_cell, self.lstm_cell_2, self.lstm_cell_3, self.lstm_cell_4, self.lstm_cell_5]

        px = self.get_embed(_input)
        hx = self.get_embed(_target)

        for model in models:
            hp, cp = model(px[0])
            p_hiddens = hp.unsqueeze(1)
            for i in range(1, px.size(0)):
                hp, cp = model(px[i], (hp, cp))
                p_hiddens = torch.cat((p_hiddens, hp.unsqueeze(1)), dim=1)

            hh, ch = model(hx[0])
            h_hiddens = hh.unsqueeze(1)
            for i in range(1, hx.size(0)):
                hh, ch = model(hx[i], (hh, ch))
                h_hiddens = torch.cat((h_hiddens, hh.unsqueeze(1)), dim=1)

            a_p, a_h = self.co_attn(p_hiddens, h_hiddens)

            px = torch.cat([px, a_p.transpose(0, 1), p_hiddens.transpose(0, 1)], dim=2)
            hx = torch.cat([hx, a_h.transpose(0, 1), h_hiddens.transpose(0, 1)], dim=2)

        return torch.max(p_hiddens, dim=1)[0], torch.max(h_hiddens, dim=1)[0]

    def dense_rnn(self, _input):
        # stacked lstm layers
        models = [self.lstm_cell, self.lstm_cell_2, self.lstm_cell_3, self.lstm_cell_4, self.lstm_cell_5]

        x = _input
        # x -> (max_length, batch_size, input_size)

        for idx, model in enumerate(models):
            hx, cx = model(x[0])
            hiddens = hx.unsqueeze(1)
            # hx -> (batch, 1, hidden_size)
            for i in range(1, x.size(0)):
                hx, cx = model(x[i], (hx, cx))
                hiddens = torch.cat((hiddens, hx.unsqueeze(1)), dim=1)

            if idx < len(models) - 1:
                hiddens = hiddens.transpose(0, 1)
                x = torch.cat((x, hiddens), dim=2)

        return torch.max(hiddens, dim=1)[0]

    def get_sent_embed(self, _input):
        embedded = self.get_embed(_input)

        # 1. encoder last hidden
        # _, (hq, cn) = self.lstm(embedded)
        # _f, _b = hq
        # return torch.cat((_f, _b), dim=1)

        # 2. word embedding avg
        # hq = torch.mean(embedded, dim=1).squeeze(1)
        # return hq

        # 3. max pooling
        # return self.max_pooling(embedded)
 
        # 3-1. self_attn + max pooling
        # hq, hs = self.self_attn(q_embedded, s_embedded)
        
        # 4. dense rnn
        return self.dense_rnn(embedded)


class BiasedLoss(nn.Module):

    def __init__(self):
        super(BiasedLoss, self).__init__()

    @staticmethod
    def forward(_input, _target):
        for i in range(_input.size(0)):
            _, idx = torch.max(_input[i], 0)

            if idx > 0 and _target[i][0] == 0:
                # unanswerable
                for j in range(_target[i].size(0)):
                    if j != idx:
                        _target[i][j] = 0

        result = torch.abs(_input * _target)
        return torch.mean(result)
