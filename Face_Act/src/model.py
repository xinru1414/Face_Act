"""
Jan 2020
Xinru Yan
"""

import torch.nn as nn
import torch
import numpy as np
from dataloader import load_pte


class RNN(nn.Module):
    def __init__(self, config, dl, device):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(num_embeddings=dl.vocab_size, embedding_dim=self.config.emb_dim,
                                  padding_idx=dl.PAD_IDX)
        # initialize word embeddings
        if config.pte_path is None:
            nn.init.xavier_uniform_(self.embedding.weight)
        else:
            emb, w2i = load_pte(config.pte_path, config.most_frequent_pte)
            init_emb = np.zeros((dl.vocab_size, config.emb_dim))
            count = 0
            for word in dl.w2i:
                if word in w2i:
                    count += 1
                    init_emb[dl.w2i[word]] = emb[w2i[word]]
            print(f'loaded {count} words from Google pte')
            self.embedding.weight.data.copy_(torch.from_numpy(init_emb))
            self.embedding.weight.requires_grad = True

        if self.config.setting == 'RNN':
            self.rnn = nn.RNN(input_size=self.config.emb_dim, hidden_size=self.config.hidden_size)
            self.fc = nn.Linear(self.config.hidden_size, self.config.output_size)
        else:
            self.rnn = nn.LSTM(input_size=self.config.emb_dim, hidden_size=self.config.hidden_size,
                               num_layers=self.config.hidden_layers,
                               bidirectional=self.config.bidirectional,
                               dropout=self.config.dropout)

            self.fc = nn.Linear(self.config.hidden_size * 2, self.config.output_size)

        self.dropout = nn.Dropout(self.config.dropout)

        self.to(device)

    def forward(self, x, seq_lengths):
        # text = [sent len, batch size]

        x = x.permute(1, 0)
        embedded = self.dropout(self.embedding(x))

        if self.config.setting == 'RNN':

            output, hidden = self.rnn(embedded)

            # output = [sent len, batch size, hid dim]
            # hidden = [1, batch size, hid dim]

            assert torch.equal(output[-1, :, :], hidden.squeeze(0))

            return self.fc(hidden.squeeze(0))
        else:
            # text = [sent len, batch size]

            embedded = self.dropout(self.embedding(x))

            # embedded = [sent len, batch size, emb dim]

            # pack sequence
            packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, seq_lengths, enforce_sorted=False)

            packed_output, (hidden, cell) = self.rnn(packed_embedded)

            # unpack sequence
            # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

            # output = [sent len, batch size, hid dim * num directions]
            # output over padding tokens are zero tensors

            # hidden = [num layers * num directions, batch size, hid dim]
            # cell = [num layers * num directions, batch size, hid dim]

            # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
            # and apply dropout

            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

            # hidden = [batch size, hid dim * num directions]

            return self.fc(hidden)

    def save(self):
        save_best_path = self.config.save_best
        torch.save(self.state_dict(), save_best_path)

    def load(self):
        load_best_path = self.config.save_best
        self.load_state_dict(torch.load(load_best_path))