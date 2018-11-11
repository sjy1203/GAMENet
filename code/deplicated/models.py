import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dnc import DNC
from layers import GraphConvolution
import scipy.sparse as sp

class GAMENet(nn.Module):
    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=64, device=torch.device('cpu:0'), with_memory=True):
        super(GAMENet, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        self.with_memory = with_memory
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.start_token = vocab_size[K-1]

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i] if i!=K-1 else vocab_size[i]+1, emb_dim)  for i in range(K)])
        self.dropout = nn.Dropout(p=0.3)

        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim * 2, batch_first=True) for _ in range(K)])
        self.query1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        self.query2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim)
        )

        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        if with_memory:
            self.output = nn.Sequential(
                nn.ReLU(),
                nn.Linear(emb_dim * 4, emb_dim * 2),
                nn.ReLU(),
                nn.Linear(emb_dim * 2, vocab_size[2])
            )
        else:
            self.output = nn.Sequential(
                nn.ReLU(),
                nn.Linear(emb_dim * 2, vocab_size[2])
            )

        self.init_weights()

    def forward(self, input, prev_input, hidden=[None] * 3):
        # input (3, codes)
        # hidden (3, gru_hidden_tensor)
        input1_hidden, input2_hidden, target_hidden = hidden

        def weighted_embedding(embedding, idx):
            # embedding (1, len, dim)
            weight = [np.exp(len(input[idx]) - i) for i in range(len(input[idx]))]
            weight = F.softmax(torch.FloatTensor(weight).view(1, -1, 1).to(self.device), dim=1) # (1, len, 1)
            weighted_embedding = weight.repeat(1,1, embedding.size(-1)) * embedding # (1, len, dim)
            return weighted_embedding.sum(dim=1).unsqueeze(dim=0) # (1, 1, dim)

        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0) # (1,1,dim)

        output1, input1_hidden = self.encoders[0](
            weighted_embedding(self.dropout(self.embeddings[0](torch.LongTensor(input[0]).unsqueeze(dim=0).to(self.device))), idx=0),
            input1_hidden
        )
        output2, input2_hidden = self.encoders[1](
            weighted_embedding(self.dropout(self.embeddings[1](torch.LongTensor(input[1]).unsqueeze(dim=0).to(self.device))), idx=1),
            input2_hidden
        )

        if prev_input is None:
            prev_input = [self.start_token]

        output3, target_hidden = self.encoders[2](
            sum_embedding(self.dropout(self.embeddings[2](torch.LongTensor(prev_input).unsqueeze(dim=0).to(self.device)))),
            target_hidden
        )

        input_hidden_combined = torch.cat([input1_hidden, input2_hidden], dim=-1).squeeze(dim=0)
        query1 = self.query1(input_hidden_combined)  # (1, dim)
        query2 = self.query2(target_hidden.squeeze(dim=0))  # (1, dim)

        if self.with_memory:
            # graph memory
            drug_memory = self.ehr_gcn() + self.ddi_gcn() * self.inter  # (size, dim)
            drug_memory_T = torch.transpose(drug_memory, 0, 1)  # (dim, size)

            key_weights1 = torch.mm(query1, drug_memory_T)  # (1, size)
            fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)

            key_weights2 = torch.mm(query2, drug_memory_T)  # (1, size)
            fact2 = torch.mm(key_weights2, drug_memory)  # (1, dim)

            output = self.output(torch.cat([query1, query2, fact1, fact2], dim=-1))
        else:
            output = self.output(torch.cat([query1, query2], dim=-1))

        if self.training:

            neg_pred_prob = F.sigmoid(output)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).sum()

            return output, [input1_hidden, input2_hidden, target_hidden], batch_neg

        else:

            return output, [input1_hidden, input2_hidden, target_hidden]

    def decode(self, logits):
        """Return probability distribution over words."""
        word_probs = F.softmax(logits, dim=-1)
        return word_probs

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

        self.inter.data.uniform_(-initrange, initrange)


class LSTM_CAT(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, device=torch.device('cpu:0')):
        super(LSTM_CAT, self).__init__()
        # K kind sub-sequence and K-1 is the target
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        # self.token_start = vocab_size[K - 1]
        # self.token_end = self.token_start + 1
        # self.output_size = self.token_start + 2

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(K)])

        self.encoders = nn.ModuleList([
            nn.LSTM(emb_dim, emb_dim * 2, batch_first=True) for i in range(K)
        ])

        self.decoder = nn.Linear(emb_dim * 2, vocab_size[K - 1])

    def forward(self, input, max_len=20):
        # input size (admission_size, K, codes)
        K = self.K
        sequence_container = []
        # decoder_point = len(input) - 1

        # input sequence

        for i in range(K - 1):
            input_k = [code for adm in input for code in adm[i]]
            x = torch.LongTensor(input_k).unsqueeze(dim=0).to(self.device)
            embed_x = self.embeddings[i](x)
            output, _ = self.encoders[i](embed_x)

            sequence_container.append(output[:, -1, :])

        # target sequence

        input_k = []
        y = np.zeros((1, self.vocab_size[K - 1]))
        for i in range(len(input)):
            if i == len(input) - 1:
                y[:, input[i][K - 1]] = 1
                break
            input_k.extend(input[i][K - 1])
        if len(input_k) != 0:
            x = torch.LongTensor(input_k).unsqueeze(dim=0).to(self.device)
            embed_x = self.embeddings[K - 1](x)
            output, _ = self.encoders[K - 1](embed_x)
            sequence_container.append(output[:, -1, :])

        context = torch.stack(sequence_container, dim=0).mean(dim=0)

        return self.decoder(context), torch.FloatTensor(y).to(self.device)

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

    def decode(self, logits):
        """Return probability distribution over words."""
        word_probs = F.softmax(logits, dim=-1)
        return word_probs


class MANN(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, device=torch.device('cpu:0')):
        super(MANN, self).__init__()
        # K kind sub-sequence and K-1 is the target
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        # self.token_start = vocab_size[K - 1]
        # self.token_end = self.token_start + 1
        # self.output_size = self.token_start + 2

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(K)])
        self.encoders = nn.ModuleList([DNC(
            input_size=emb_dim,
            hidden_size=emb_dim * 2,
            rnn_type='lstm',
            num_layers=1,
            num_hidden_layers=1,
            nr_cells=16,
            cell_size=emb_dim,
            read_heads=1,
            batch_first=True,
            gpu_id=0
        ) for _ in range(K)])

        self.decoder = nn.Linear(emb_dim, vocab_size[K - 1])

    def forward(self, input, max_len=20):
        # input size (admission_size, K, codes)
        K = self.K
        sequence_container = []
        # decoder_point = len(input) - 1

        # input sequence

        for i in range(K - 1):
            input_k = [code for adm in input for code in adm[i]]
            x = torch.LongTensor(input_k).unsqueeze(dim=0).to(self.device)
            embed_x = self.embeddings[i](x)
            output, (controller_hidden, memory, read_vectors) = self.encoders[i](embed_x, reset_experience=True)
            # output, _ = self.encoders[i](embed_x)

            sequence_container.append(output[:, -1, :])

            # target sequence

        input_k = []
        y = np.zeros((1, self.vocab_size[K - 1]))
        for i in range(len(input)):
            if i == len(input) - 1:
                y[:, input[i][K - 1]] = 1
                break
            input_k.extend(input[i][K - 1])
        if len(input_k) != 0:
            x = torch.LongTensor(input_k).unsqueeze(dim=0).to(self.device)
            embed_x = self.embeddings[K - 1](x)
            output, (controller_hidden, memory, read_vectors) = self.encoders[K - 1](embed_x, reset_experience=True)

            sequence_container.append(output[:, -1, :])

        context = torch.stack(sequence_container, dim=0).mean(dim=0)

        return self.decoder(context), torch.FloatTensor(y).to(self.device)

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

    def decode(self, logits):
        """Return probability distribution over words."""
        word_probs = F.softmax(logits, dim=-1)
        return word_probs


class CycleRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, device=torch.device('cpu:0')):
        super(CycleRNN, self).__init__()
        # K kind sub-sequence and K-1 is the target
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        self.token_start = vocab_size[K - 1]
        self.token_end = self.token_start + 1
        self.output_size = self.token_start + 2

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i] if i != K - 1 else self.output_size, emb_dim) for i in range(K)])
        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim, batch_first=True) for i in range(K)])

        self.decoder = nn.GRU(input_size=emb_dim, hidden_size=emb_dim, batch_first=True)

        self.output = nn.Linear(emb_dim, self.output_size)

        self.init_weights()

    def forward(self, input, max_len=20):
        # input size (admission_size, K, codes)
        K = self.K
        decoder_point = len(input) - 1

        state_container = [None] * K
        for i in range(len(input)):
            for j in range(K):
                if i == decoder_point and j == K - 1:
                    break

                x = torch.LongTensor(input[i][j]).unsqueeze(dim=0).to(self.device)

                embed_x = self.embeddings[j](x)
                _, h_n = self.encoders[j](embed_x, state_container[(j - 1) % K])
                state_container[j] = h_n

        x = input[-1][-1]
        if state_container[K - 1] is None:
            # visit < 2
            encoder_init_state = torch.stack(state_container[:K - 1], dim=0).mean(dim=0)
        else:
            encoder_init_state = torch.stack(state_container, dim=0).mean(dim=0)

        if self.training:
            # teacher force encoder

            enc_input = [self.token_start] + x

            embed_x = self.embeddings[K - 1](torch.LongTensor(enc_input).unsqueeze(dim=0).to(self.device))
            output, _ = self.decoder(embed_x, encoder_init_state)
            output = output.squeeze(dim=0)
            return self.output(output), torch.LongTensor(x + [self.token_end]).to(self.device)

        else:
            # student force encoder
            y_gt = np.zeros(self.vocab_size[K - 1])
            y_pred_prob = y_gt.copy()
            y_pred = y_gt.copy()
            y_gt[x] = 1

            enc_input = torch.LongTensor([self.token_start]).unsqueeze(dim=0).to(self.device)
            for _ in range(max_len):
                embed_x = self.embeddings[K - 1](enc_input)
                output, encoder_init_state = self.decoder(embed_x, encoder_init_state)
                output = output.squeeze(dim=0)
                output = self.decode(self.output(output).squeeze())
                output, argmax = torch.max(output, dim=-1)
                enc_input = argmax.unsqueeze(dim=0).unsqueeze(dim=0)

                if argmax.item() == self.token_end:
                    break
                if argmax.item() == self.token_start:
                    continue

                y_pred[argmax.item()] = 1
                y_pred_prob[argmax.item()] = output.item()

            return y_pred, y_pred_prob, y_gt

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

    def decode(self, logits):
        """Return probability distribution over words."""
        word_probs = F.softmax(logits, dim=-1)
        return word_probs


class MANN_Attn(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, device=torch.device('cpu:0')):
        super(MANN_Attn, self).__init__()
        # K kind sub-sequence and K-1 is the target
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device

        self.token_start = vocab_size[K - 1]
        self.token_split = self.token_start + 1
        self.token_end = self.token_split + 1

        self.output_size = self.token_end + 1

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i] if i != K - 1 else self.output_size, emb_dim) for i in range(K)])
        self.dropout = nn.Dropout()

        self.encoders = nn.ModuleList([DNC(
            input_size=emb_dim,
            hidden_size=emb_dim,
            rnn_type='gru',
            num_layers=1,
            num_hidden_layers=1,
            nr_cells=16,
            cell_size=emb_dim,
            read_heads=1,
            batch_first=True,
            gpu_id=0
        ) for _ in range(K - 1)])
        self.attn = nn.ModuleList([nn.Linear(emb_dim * 3, 1, bias=False) for _ in range(K - 1)])

        self.decoder = DNC(
            input_size=emb_dim * 2,
            hidden_size=emb_dim * 2,
            rnn_type='gru',
            num_layers=1,
            num_hidden_layers=1,
            nr_cells=16,
            cell_size=emb_dim * 2,
            read_heads=1,
            batch_first=True,
            gpu_id=0
        )
        self.attn_combined = nn.Linear(emb_dim * 3, emb_dim * 2)

        self.output = nn.Linear(emb_dim * 4, self.output_size)

        self.init_weights()

    def attention(self, h, o, i):
        # h (1, 1, size) o (1, seq, size)
        h = h.repeat(1, o.size(1), 1)
        attn_weight = F.softmax(self.attn[i](torch.cat([h, o], dim=-1)).squeeze(dim=-1), dim=-1).unsqueeze(
            dim=0)  # (1, 1 ,seq)
        return torch.bmm(attn_weight, o)  # (1,1, size)

    def forward(self, input):

        # input size (admission_size, K, codes)
        assert self.K == 3
        pre_tensor_outputs = []
        pre_target_label = []

        last_tensor_outputs = []
        last_target_label = []

        input1_hidden, memory1, read_vectors1, reset_memory = None, None, None, True
        input2_hidden, memory2, read_vectors2, reset_memory = None, None, None, True
        target_hidden, memory3, read_vectors3, reset_memory = None, None, None, True
        for i in range(len(input)):
            output1, (input1_hidden, memory1, read_vectors1) = self.encoders[0](
                self.dropout(self.embeddings[0](torch.LongTensor(input[i][0]).unsqueeze(dim=0).to(self.device))),
                (input1_hidden, memory1, read_vectors1),
                reset_experience=reset_memory
            )

            output2, (input2_hidden, memory2, read_vectors2) = self.encoders[1](
                self.dropout(self.embeddings[1](torch.LongTensor(input[i][1]).unsqueeze(dim=0).to(self.device))),
                (input2_hidden, memory2, read_vectors2),
                reset_experience=reset_memory
            )

            # pre admission
            # input start token or split token first depend on i
            target_token_sequences = [self.token_start] if i == 0 else [self.token_split]
            target_token_sequences += input[i][2]

            pre_target_label.extend(input[i][2])
            pre_target_label.append(self.token_split if i != len(input) - 1 else self.token_end)

            if i == len(input) - 1:
                last_target_label.extend(input[i][2])
                last_target_label.append(self.token_end)

            if target_hidden is None:
                target_hidden = [torch.cat(input1_hidden + input2_hidden, dim=-1)]
            predict_idx = None
            for input_token in target_token_sequences:
                if not self.training and not predict_idx is None:
                    input_token = predict_idx.item()

                target_input = self.dropout(
                    self.embeddings[2](torch.LongTensor([input_token]).unsqueeze(dim=0).to(self.device)))

                input1_context = self.attention(target_hidden[0], output1, 0)
                input2_context = self.attention(target_hidden[0], output2, 1)

                combined = self.attn_combined(torch.cat([target_input, input1_context, input2_context], dim=-1))

                target_output, (target_hidden, memory3, read_vectors3) = self.decoder(F.relu(combined),
                                                                                      (target_hidden, memory3,
                                                                                       read_vectors3),
                                                                                      reset_experience=reset_memory
                                                                                      )

                output = self.output(torch.cat([target_hidden[0], input1_context, input2_context], dim=-1)).squeeze(
                    dim=0)
                predict_idx = torch.argmax(output, dim=-1)

                pre_tensor_outputs.append(output)

                if i == len(input) - 1:
                    last_tensor_outputs.append(output)

                reset_memory = False

        return torch.cat(pre_tensor_outputs, dim=0), \
               torch.LongTensor(pre_target_label).to(self.device), \
               torch.cat(last_tensor_outputs, dim=0), \
               torch.LongTensor(last_target_label).to(self.device)

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

    def decode(self, logits):
        """Return probability distribution over words."""
        word_probs = F.softmax(logits, dim=-1)
        return word_probs


class RNN_Attn(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, device=torch.device('cpu:0')):
        super(RNN_Attn, self).__init__()
        # K kind sub-sequence and K-1 is the target
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device

        self.token_start = vocab_size[K - 1]
        self.token_split = self.token_start + 1
        self.token_end = self.token_split + 1

        self.output_size = self.token_end + 1

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i] if i != K - 1 else self.output_size, emb_dim) for i in range(K)])
        self.dropout = nn.Dropout()

        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim, batch_first=True) for i in range(K)])
        self.attn = nn.ModuleList([nn.Linear(emb_dim * 3, 1, bias=False) for _ in range(K - 1)])

        self.decoder = nn.GRU(input_size=emb_dim * 2, hidden_size=emb_dim * 2, batch_first=True)
        self.attn_combined = nn.Linear(emb_dim * 3, emb_dim * 2)

        self.output = nn.Linear(emb_dim * 4, self.output_size)

        self.init_weights()

    def attention(self, h, o, i):
        # h (1, 1, size) o (1, seq, size)
        h = h.repeat(1, o.size(1), 1)
        attn_weight = F.softmax(self.attn[i](torch.cat([h, o], dim=-1)).squeeze(dim=-1), dim=-1).unsqueeze(
            dim=0)  # (1, 1 ,seq)
        return torch.bmm(attn_weight, o)  # (1,1, size)

    def forward(self, input):

        # input size (admission_size, K, codes)
        assert self.K == 3
        pre_tensor_outputs = []
        pre_target_label = []

        last_tensor_outputs = []
        last_target_label = []

        input1_hidden = None
        input2_hidden = None
        target_hidden = None
        for i in range(len(input)):
            output1, input1_hidden = self.encoders[0](
                self.dropout(self.embeddings[0](torch.LongTensor(input[i][0]).unsqueeze(dim=0).to(self.device))),
                input1_hidden
            )
            output2, input2_hidden = self.encoders[1](
                self.dropout(self.embeddings[1](torch.LongTensor(input[i][1]).unsqueeze(dim=0).to(self.device))),
                input2_hidden
            )
            if target_hidden is None:
                target_hidden = torch.cat([input1_hidden, input2_hidden], dim=-1)

            # pre admission
            # input start token or split token first depend on i
            target_token_sequences = [self.token_start] if i == 0 else [self.token_split]
            target_token_sequences += input[i][2]

            pre_target_label.extend(input[i][2])
            pre_target_label.append(self.token_split if i != len(input) - 1 else self.token_end)

            if i == len(input) - 1:
                last_target_label.extend(input[i][2])
                last_target_label.append(self.token_end)

            predict_idx = None
            for input_token in target_token_sequences:
                if not self.training and not predict_idx is None:
                    input_token = predict_idx.item()

                target_input = self.dropout(
                    self.embeddings[2](torch.LongTensor([input_token]).unsqueeze(dim=0).to(self.device)))

                input1_context = self.attention(target_hidden, output1, 0)
                input2_context = self.attention(target_hidden, output2, 1)

                combined = self.attn_combined(torch.cat([target_input, input1_context, input2_context], dim=-1))

                target_output, target_hidden = self.decoder(F.relu(combined), target_hidden)

                output = self.output(torch.cat([target_hidden, input1_context, input2_context], dim=-1)).squeeze(dim=0)
                predict_idx = torch.argmax(output, dim=-1)

                pre_tensor_outputs.append(output)

                if i == len(input) - 1:
                    last_tensor_outputs.append(output)

        return torch.cat(pre_tensor_outputs, dim=0), \
               torch.LongTensor(pre_target_label).to(self.device), \
               torch.cat(last_tensor_outputs, dim=0), \
               torch.LongTensor(last_target_label).to(self.device)

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

    def decode(self, logits):
        """Return probability distribution over words."""
        word_probs = F.softmax(logits, dim=-1)
        return word_probs


class RNN_Two(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, device=torch.device('cpu:0')):
        super(RNN_Two, self).__init__()
        K = len(vocab_size)
        assert K == 3
        self.K = K
        self.vocab_size = vocab_size
        self.device = device

        self.token_start = vocab_size[2]
        self.token_end = self.token_start + 1
        self.output_size = self.token_end + 1

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i] if i != K - 1 else self.output_size, emb_dim) for i in range(K)])
        self.dropout = nn.Dropout()

        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim * 2, batch_first=True) for _ in range(K - 1)])

        self.mlp_decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim * 2, out_features=vocab_size[2])
        )
        self.rnn_decoder = nn.GRU(input_size=emb_dim, hidden_size=emb_dim * 2, batch_first=True)
        self.rnn_output = nn.Linear(emb_dim * 2, self.output_size)

    def forward(self, input, hidden=[None] * 3):
        # input (3, codes)
        # hidden (3, gru_hidden_tensor)
        input1_hidden, input2_hidden, target_hidden = hidden
        output1, input1_hidden = self.encoders[0](
            self.dropout(self.embeddings[0](torch.LongTensor(input[0]).unsqueeze(dim=0).to(self.device))),
            input1_hidden
        )
        output2, input2_hidden = self.encoders[1](
            self.dropout(self.embeddings[1](torch.LongTensor(input[1]).unsqueeze(dim=0).to(self.device))),
            input2_hidden
        )
        input_hidden_combined = torch.cat([input1_hidden, input2_hidden], dim=-1).squeeze(dim=0)

        # mlp predict
        target_output1 = self.mlp_decoder(input_hidden_combined)  # (1, vocab_size)

        # rnn predict
        target_token_sequences = [self.token_start] + input[2]
        target_output2, target_hidden = self.rnn_decoder(
            self.dropout(self.embeddings[2](torch.LongTensor(target_token_sequences).unsqueeze(dim=0).to(self.device))),
            target_hidden
        )
        target_output2 = self.rnn_output(target_output2.squeeze(dim=0))

        return target_output1, target_output2, [input1_hidden, input2_hidden, target_hidden]

    def seq_evaluate(self, target_hidden, max_len=15):
        # input (codes)
        # hidden (gru_hidden_tensor)
        output_logits = []
        output_labels = []
        input_token = self.token_start
        for i in range(max_len):
            target_embed = self.embeddings[2](torch.LongTensor([input_token]).unsqueeze(dim=0).to(self.device))
            target_input = self.dropout(target_embed)

            target_output, target_hidden = self.rnn_decoder(target_input, target_hidden)
            target_output = self.rnn_output(target_output.squeeze(dim=0)).squeeze(dim=0)

            input_token = torch.argmax(target_output, dim=-1)
            input_token = input_token.item()
            if input_token == self.token_end:
                break

            output_logits.append(self.decode(target_output).detach().cpu().numpy())
            output_labels.append(input_token)

        return output_logits, output_labels

    def decode(self, logits):
        """Return probability distribution over words."""
        word_probs = F.softmax(logits, dim=-1)
        return word_probs


class Temp(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, device=torch.device('cpu:0')):
        super(Temp, self).__init__()
        # K kind sub-sequence and K-1 is the target
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device

        self.token_start = vocab_size[K - 1]
        self.token_split = self.token_start + 1
        self.token_end = self.token_split + 1

        self.output_size = self.token_end + 1

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i] if i != K - 1 else self.output_size, emb_dim) for i in range(K)])
        self.dropout = nn.Dropout()

        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim, batch_first=True) for i in range(K)])
        self.attn = nn.ModuleList([nn.Linear(emb_dim * 3, 1, bias=False) for _ in range(K - 1)])

        self.decoder = nn.GRU(input_size=emb_dim * 2, hidden_size=emb_dim * 2, batch_first=True)
        self.attn_combined = nn.Linear(emb_dim * 3, emb_dim * 2)

        self.output = nn.Linear(emb_dim * 4, self.output_size)

        self.init_weights()

    def attention(self, h, o, i):
        # h (1, 1, size) o (1, seq, size)
        h = h.repeat(1, o.size(1), 1)
        attn_weight = F.softmax(self.attn[i](torch.cat([h, o], dim=-1)).squeeze(dim=-1), dim=-1).unsqueeze(
            dim=0)  # (1, 1 ,seq)
        return torch.bmm(attn_weight, o)  # (1,1, size)

    def forward(self, input):

        # input size (admission_size, K, codes)
        assert self.K == 3
        pre_tensor_outputs = []
        pre_target_label = []

        last_tensor_outputs = []
        last_target_label = []

        input1_hidden = None
        input2_hidden = None
        target_hidden = None
        for i in range(len(input)):
            output1, input1_hidden = self.encoders[0](
                self.dropout(self.embeddings[0](torch.LongTensor(input[i][0]).unsqueeze(dim=0).to(self.device))),
                input1_hidden
            )
            output2, input2_hidden = self.encoders[1](
                self.dropout(self.embeddings[1](torch.LongTensor(input[i][1]).unsqueeze(dim=0).to(self.device))),
                input2_hidden
            )
            if target_hidden is None:
                target_hidden = torch.cat([input1_hidden, input2_hidden], dim=-1)

            # pre admission
            # input start token or split token first depend on i
            target_token_sequences = [self.token_start] if i == 0 else [self.token_split]
            target_token_sequences += input[i][2]

            pre_target_label.extend(input[i][2])
            pre_target_label.append(self.token_split if i != len(input) - 1 else self.token_end)

            if i == len(input) - 1:
                last_target_label.extend(input[i][2])
                last_target_label.append(self.token_end)

            predict_idx = None
            for input_token in target_token_sequences:
                if not self.training and not predict_idx is None:
                    input_token = predict_idx.item()

                target_input = self.dropout(
                    self.embeddings[2](torch.LongTensor([input_token]).unsqueeze(dim=0).to(self.device)))

                input1_context = self.attention(target_hidden, output1, 0)
                input2_context = self.attention(target_hidden, output2, 1)

                combined = self.attn_combined(torch.cat([target_input, input1_context, input2_context], dim=-1))

                target_output, target_hidden = self.decoder(F.relu(combined), target_hidden)

                output = self.output(torch.cat([target_hidden, input1_context, input2_context], dim=-1)).squeeze(dim=0)
                predict_idx = torch.argmax(output, dim=-1)

                pre_tensor_outputs.append(output)

                if i == len(input) - 1:
                    last_tensor_outputs.append(output)

        return torch.cat(pre_tensor_outputs, dim=0), \
               torch.LongTensor(pre_target_label).to(self.device), \
               torch.cat(last_tensor_outputs, dim=0), \
               torch.LongTensor(last_target_label).to(self.device)

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

    def decode(self, logits):
        """Return probability distribution over words."""
        word_probs = F.softmax(logits, dim=-1)
        return word_probs


class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class GMNN(nn.Module):
    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=64, device=torch.device('cpu:0')):
        super(GMNN, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(K - 1)])
        self.dropout = nn.Dropout(p=0.3)

        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim * 2, batch_first=True) for _ in range(K - 1)])
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim)
        )

        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(emb_dim , emb_dim))

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2])
        )

        ## seq predict
        self.seq_encoder = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.seq_output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2])
        )

        self.init_weights()

    def forward(self, input, hidden=[None] * 3):
        # input (3, codes)
        # hidden (3, gru_hidden_tensor)
        input1_hidden, input2_hidden, target_hidden = hidden
        output1, input1_hidden = self.encoders[0](
            self.dropout(self.embeddings[0](torch.LongTensor(input[0]).unsqueeze(dim=0).to(self.device))),
            input1_hidden
        )
        output2, input2_hidden = self.encoders[1](
            self.dropout(self.embeddings[1](torch.LongTensor(input[1]).unsqueeze(dim=0).to(self.device))),
            input2_hidden
        )
        input_hidden_combined = torch.cat([input1_hidden, input2_hidden], dim=-1).squeeze(dim=0)
        query = self.query(input_hidden_combined)  # (1, dim)

        # structure memory
        drug_memory = self.ehr_gcn() + torch.mm(self.ddi_gcn(), self.inter)  # (size, dim)
        drug_memory_T = torch.transpose(drug_memory, 0, 1)  # (dim, size)
        key_weights = torch.mm(query, drug_memory_T)  # (1, size)
        fact = torch.mm(key_weights, drug_memory)  # (1, dim)
        output = self.output(torch.cat([fact, query], dim=-1))

        if self.training:
            true_embedding = torch.index_select(drug_memory, dim=0, index=torch.LongTensor(input[2]).to(self.device))  # (seq, codes)
            # neg pos batch loss
            # pos
            batch_pos = torch.mm(true_embedding, true_embedding.t())
            batch_pos = -F.logsigmoid(batch_pos).sum()
            # neg
            all_neg = torch.mm(drug_memory, drug_memory.t())
            all_neg = -F.logsigmoid(-all_neg)
            batch_neg = all_neg.mul(self.tensor_ddi_adj)
            neg_pred_prob = F.sigmoid(output)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
            batch_neg = neg_pred_prob.mul(batch_neg).sum()

            # seq predict
            seq_input = true_embedding
            seq_input = torch.cat([fact, seq_input], dim=0).unsqueeze(dim=0)  # (1, seq+1, codes)
            o, h = self.seq_encoder(seq_input, query.unsqueeze(dim=0) if target_hidden is None else target_hidden)
            seq_output = self.seq_output(o.squeeze(dim=0))
            return output, seq_output, [input1_hidden, input2_hidden, h], batch_pos, batch_neg
        else:
            output_logits, output_labels, target_hidden = self.seq_evaluate(query.unsqueeze(dim=0),
                                                                            fact.unsqueeze(dim=0))
            return output, output_logits, output_labels, [input1_hidden, input2_hidden, target_hidden]

    def seq_evaluate(self, target_hidden, target_input, max_len=10):
        # input (codes)
        # hidden (gru_hidden_tensor)
        output_logits = []
        output_labels = []
        drug_memory = self.ehr_gcn() + torch.mm(self.ddi_gcn(), self.inter)  # (size, dim)

        for i in range(max_len):
            target_output, target_hidden = self.seq_encoder(target_input, target_hidden)
            target_output = self.seq_output(target_output.squeeze(dim=0)).squeeze(dim=0)

            input_token = torch.argmax(target_output, dim=-1)
            input_token = input_token.item()

            target_input = torch.index_select(drug_memory, dim=0, index=torch.LongTensor([input_token]).to(self.device)).unsqueeze(dim=0) # (1, code)

            output_logits.append(self.decode(target_output).detach().cpu().numpy())
            output_labels.append(input_token)

        return output_logits, output_labels, target_hidden

    def decode(self, logits):
        """Return probability distribution over words."""
        word_probs = F.softmax(logits, dim=-1)
        return word_probs

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

        self.inter.data.uniform_(-initrange,initrange)


class SMM(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, device=torch.device('cpu:0')):
        super(SMM, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(K)])
        self.dropout = nn.Dropout(p=0.3)

        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim * 2, batch_first=True) for _ in range(K - 1)])
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim)
        )

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2])
        )

        ## seq predict
        self.seq_encoder = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.seq_output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2])
        )

        self.init_weights()

    def forward(self, input, hidden=[None] * 3):
        # input (3, codes)
        # hidden (3, gru_hidden_tensor)
        input1_hidden, input2_hidden, target_hidden = hidden
        output1, input1_hidden = self.encoders[0](
            self.dropout(self.embeddings[0](torch.LongTensor(input[0]).unsqueeze(dim=0).to(self.device))),
            input1_hidden
        )
        output2, input2_hidden = self.encoders[1](
            self.dropout(self.embeddings[1](torch.LongTensor(input[1]).unsqueeze(dim=0).to(self.device))),
            input2_hidden
        )
        input_hidden_combined = torch.cat([input1_hidden, input2_hidden], dim=-1).squeeze(dim=0)
        query = self.query(input_hidden_combined)  # (1, dim)
        drug_memory = self.dropout(self.embeddings[2](
            torch.LongTensor(range(0, self.vocab_size[2])).unsqueeze(dim=0).to(self.device))).squeeze(
            dim=0)  # (size, dim)
        drug_memory_T = torch.transpose(drug_memory, 0, 1)  # (dim, size)
        key_weights = torch.mm(query, drug_memory_T)  # (1, size)
        fact = torch.mm(key_weights, drug_memory)  # (1, dim)
        output = self.output(torch.cat([fact, query], dim=-1))

        # seq predict
        if self.training:
            seq_input = self.dropout(
                self.embeddings[2](torch.LongTensor(input[2]).unsqueeze(dim=0).to(self.device)))  # (1, seq, codes)
            seq_input = torch.cat([fact.unsqueeze(dim=0), seq_input], dim=1)  # (1, seq+1, codes)
            o, h = self.seq_encoder(seq_input, query.unsqueeze(dim=0) if target_hidden is None else target_hidden)
            seq_output = self.seq_output(o.squeeze(dim=0))
            return output, seq_output, [input1_hidden, input2_hidden, h]
        else:
            output_logits, output_labels, target_hidden = self.seq_evaluate(query.unsqueeze(dim=0),
                                                                            fact.unsqueeze(dim=0))
            return output, output_logits, output_labels, [input1_hidden, input2_hidden, target_hidden]

    def seq_evaluate(self, target_hidden, target_input, max_len=10):
        # input (codes)
        # hidden (gru_hidden_tensor)
        output_logits = []
        output_labels = []

        for i in range(max_len):
            target_output, target_hidden = self.seq_encoder(target_input, target_hidden)
            target_output = self.seq_output(target_output.squeeze(dim=0)).squeeze(dim=0)

            input_token = torch.argmax(target_output, dim=-1)
            input_token = input_token.item()

            target_input = self.embeddings[2](torch.LongTensor([input_token]).unsqueeze(dim=0).to(self.device))

            output_logits.append(self.decode(target_output).detach().cpu().numpy())
            output_labels.append(input_token)

        return output_logits, output_labels, target_hidden

    def decode(self, logits):
        """Return probability distribution over words."""
        word_probs = F.softmax(logits, dim=-1)
        return word_probs

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)


class MA_DDNC(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, device=torch.device('cpu:0')):
        super(MA_DDNC, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device

        self.token_start = vocab_size[2]
        self.token_end = vocab_size[2] + 1

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i] if i != 2 else vocab_size[2] + 2, emb_dim) for i in range(K)])

        self.encoders = nn.ModuleList([DNC(
            input_size=emb_dim,
            hidden_size=emb_dim,
            rnn_type='gru',
            num_layers=1,
            num_hidden_layers=1,
            nr_cells=16,
            cell_size=emb_dim,
            read_heads=1,
            batch_first=True,
            gpu_id=0
        ) for _ in range(K - 1)])

        self.decoder = nn.GRU(emb_dim + 2 * emb_dim, )


class DMNC(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, device=torch.device('cpu:0')):
        super(DMNC, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device

        self.token_start = vocab_size[2]
        self.token_end = vocab_size[2] + 1

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i] if i != 2 else vocab_size[2] + 2, emb_dim) for i in range(K)])
        self.dropout = nn.Dropout(p=0.3)

        self.encoders = nn.ModuleList([DNC(
            input_size=emb_dim,
            hidden_size=emb_dim,
            rnn_type='gru',
            num_layers=1,
            num_hidden_layers=1,
            nr_cells=16,
            cell_size=emb_dim,
            read_heads=1,
            batch_first=True,
            gpu_id=0,
            independent_linears=False
        ) for _ in range(K - 1)])

        self.decoder = nn.GRU(emb_dim + emb_dim * 2, emb_dim * 2,
                              batch_first=True)  # input: (y, r1, r2,) hidden: (hidden1, hidden2)
        self.interface_weighting = nn.Linear(emb_dim * 2, 2 * (emb_dim + 1 + 3))  # 2 read head (key, str, mode)
        self.decoder_r2o = nn.Linear(2 * emb_dim, emb_dim * 2)

        self.output = nn.Linear(emb_dim * 2, vocab_size[2] + 2)

    def forward(self, input, i1_state=None, i2_state=None, h_n=None, max_len=15):
        # input (3, code)
        i1_input_tensor = self.embeddings[0](
            torch.LongTensor(input[0]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)
        i2_input_tensor = self.embeddings[1](
            torch.LongTensor(input[1]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)

        o1, (ch1, m1, r1) = \
            self.encoders[0](i1_input_tensor, (None, None, None) if i1_state is None else i1_state)
        o2, (ch2, m2, r2) = \
            self.encoders[1](i2_input_tensor, (None, None, None) if i2_state is None else i2_state)

        # save memory state
        i1_state = (ch1, m1, r1)
        i2_state = (ch2, m2, r2)

        predict_sequence = [self.token_start] + input[2]
        if h_n is None:
            h_n = torch.cat([ch1[0], ch2[0]], dim=-1)

        output_logits = []
        r1 = r1.unsqueeze(dim=0)
        r2 = r2.unsqueeze(dim=0)

        if self.training:
            for item in predict_sequence:
                # teacher force predict drug
                item_tensor = self.embeddings[2](
                    torch.LongTensor([item]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)

                o3, h_n = self.decoder(torch.cat([item_tensor, r1, r2], dim=-1), h_n)
                read_keys, read_strengths, read_modes = self.decode_read_variable(h_n.squeeze(0))

                # read from i1_mem, i2_mem and i3_mem
                r1, _ = self.read_from_memory(self.encoders[0],
                                              read_keys[:, 0, :].unsqueeze(dim=1),
                                              read_strengths[:, 0].unsqueeze(dim=1),
                                              read_modes[:, 0, :].unsqueeze(dim=1), i1_state[1])

                r2, _ = self.read_from_memory(self.encoders[1],
                                              read_keys[:, 1, :].unsqueeze(dim=1),
                                              read_strengths[:, 1].unsqueeze(dim=1),
                                              read_modes[:, 1, :].unsqueeze(dim=1), i2_state[1])

                output = self.decoder_r2o(torch.cat([r1, r2], dim=-1))
                output = self.output(output + o3).squeeze(dim=0)
                output_logits.append(output)
        else:
            item_tensor = self.embeddings[2](
                torch.LongTensor([self.token_start]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)
            for idx in range(max_len):
                # predict
                # teacher force predict drug
                o3, h_n = self.decoder(torch.cat([item_tensor, r1, r2], dim=-1), h_n)
                read_keys, read_strengths, read_modes = self.decode_read_variable(h_n.squeeze(0))

                # read from i1_mem, i2_mem and i3_mem
                r1, _ = self.read_from_memory(self.encoders[0],
                                              read_keys[:, 0, :].unsqueeze(dim=1),
                                              read_strengths[:, 0].unsqueeze(dim=1),
                                              read_modes[:, 0, :].unsqueeze(dim=1), i1_state[1])

                r2, _ = self.read_from_memory(self.encoders[1],
                                              read_keys[:, 1, :].unsqueeze(dim=1),
                                              read_strengths[:, 1].unsqueeze(dim=1),
                                              read_modes[:, 1, :].unsqueeze(dim=1), i2_state[1])

                output = self.decoder_r2o(torch.cat([r1, r2], dim=-1))
                output = self.output(output + o3).squeeze(dim=0)
                output = F.softmax(output, dim=-1)
                output_logits.append(output)

                input_token = torch.argmax(output, dim=-1)
                input_token = input_token.item()
                item_tensor = self.embeddings[2](
                    torch.LongTensor([input_token]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)

        return torch.cat(output_logits, dim=0), i1_state, i2_state, h_n

    def read_from_memory(self, dnc, read_key, read_str, read_mode, m_hidden):
        read_vectors, hidden = dnc.memories[0].read(read_key, read_str, read_mode, m_hidden)
        return read_vectors, hidden

    def decode_read_variable(self, input):
        w = 64
        r = 2
        b = input.size(0)

        input = self.interface_weighting(input)
        # r read keys (b * w * r)
        read_keys = F.tanh(input[:, :r * w].contiguous().view(b, r, w))
        # r read strengths (b * r)
        read_strengths = F.softplus(input[:, r * w:r * w + r].contiguous().view(b, r))
        # read modes (b * 3*r)
        read_modes = F.softmax(input[:, (r * w + r):].contiguous().view(b, r, 3), -1)
        return read_keys, read_strengths, read_modes


class MA_DNC(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, device=torch.device('cpu:0')):
        super(MA_DNC, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device

        self.token_start = vocab_size[2]
        self.token_end = vocab_size[2] + 1

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i] if i != 2 else vocab_size[2] + 2, emb_dim) for i in range(K)])
        self.dropout = nn.Dropout(p=0.3)

        self.encoders = nn.ModuleList([DNC(
            input_size=emb_dim,
            hidden_size=emb_dim,
            rnn_type='gru',
            num_layers=1,
            num_hidden_layers=1,
            nr_cells=16,
            cell_size=emb_dim,
            read_heads=1,
            batch_first=True,
            gpu_id=0
        ) for _ in range(K)])

        self.decoder = nn.GRU(emb_dim * 4, emb_dim * 3, batch_first=True)  # input: (y, r1, r2, r3,) hidden: (o1,o2,o3)

        self.decoder_r2o = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 3),
            nn.ReLU()
        )
        self.output = nn.Linear(emb_dim * 3, vocab_size[2] + 2)

    def forward(self, input, i1_state=None, i2_state=None, i3_state=None, max_len=15):
        # input (3, code)
        i1_input_tensor = self.dropout(
            self.embeddings[0](torch.LongTensor(input[0]).unsqueeze(dim=0).to(self.device)))  # (1, seq, codes)
        i2_input_tensor = self.dropout(
            self.embeddings[1](torch.LongTensor(input[1]).unsqueeze(dim=0).to(self.device)))  # (1, seq, codes)

        o1, (ch1, m1, r1) = \
            self.encoders[0](i1_input_tensor, (None, None, None) if i1_state is None else i1_state)
        o2, (ch2, m2, r2) = \
            self.encoders[1](i2_input_tensor, (None, None, None) if i2_state is None else i2_state)

        # save memory state
        i1_state = (ch1, m1, r1)
        i2_state = (ch2, m2, r2)

        predict_sequence = [self.token_start] + input[2]
        if i3_state is None:
            h_n = None
            token_start_tensor = self.dropout(
                self.embeddings[2](
                    torch.LongTensor([self.token_start]).unsqueeze(dim=0).to(self.device)))  # (1, seq, codes)
            o3, (ch3, m3, r3) = \
                self.encoders[2](token_start_tensor, (None, None, None))

        else:
            (ch3, m3, r3) = i3_state
            h_n = torch.cat([ch1[0], ch2[0], ch3[0]], dim=-1)

        output_logits = []

        if self.training:
            for item in predict_sequence:
                # teacher force predict drug
                item_tensor = self.dropout(
                    self.embeddings[2](torch.LongTensor([item]).unsqueeze(dim=0).to(self.device)))  # (1, seq, codes)

                _, h_n = self.decoder(
                    torch.cat([item_tensor, r1.unsqueeze(dim=0), r2.unsqueeze(dim=0), r3.unsqueeze(dim=0)], dim=-1),
                    h_n)
                o_tensor = h_n.view(1, 1, 3, -1)
                o1, o2, o3 = o_tensor[:, :, 0, :], o_tensor[:, :, 1, :], o_tensor[:, :, 2, :]

                # read from i1_mem, i2_mem and i3_mem
                o1, (ch1, m1, r1) = \
                    self.encoders[0](o1, (None, None, None) if i1_state is None else i1_state)
                o2, (ch2, m2, r2) = \
                    self.encoders[1](o2, (None, None, None) if i2_state is None else i2_state)
                o3, (ch3, m3, r3) = \
                    self.encoders[2](o3, (None, None, None) if i3_state is None else i3_state)

                output = self.decoder_r2o(torch.cat([r1, r2, r3], dim=-1))
                output = self.output(torch.cat([o1, o2, o3], dim=-1).squeeze(dim=0) + output)
                output_logits.append(output)
        else:
            item_tensor = self.dropout(
                self.embeddings[2](
                    torch.LongTensor([self.token_start]).unsqueeze(dim=0).to(self.device)))  # (1, seq, codes)
            for idx in range(max_len):
                # predict
                # teacher force predict drug
                _, h_n = self.decoder(
                    torch.cat([item_tensor, r1.unsqueeze(dim=0), r2.unsqueeze(dim=0), r3.unsqueeze(dim=0)], dim=-1),
                    h_n)
                o_tensor = h_n.view(1, 1, 3, -1)
                o1, o2, o3 = o_tensor[:, :, 0, :], o_tensor[:, :, 1, :], o_tensor[:, :, 2, :]

                # read from i1_mem, i2_mem and i3_mem
                o1, (ch1, m1, r1) = \
                    self.encoders[0](o1, (None, None, None) if i1_state is None else i1_state)
                o2, (ch2, m2, r2) = \
                    self.encoders[1](o2, (None, None, None) if i2_state is None else i2_state)
                o3, (ch3, m3, r3) = \
                    self.encoders[2](o3, (None, None, None) if i3_state is None else i3_state)

                output = self.decoder_r2o(torch.cat([r1, r2, r3], dim=-1).squeeze(dim=0))
                output = self.output(torch.cat([o1, o2, o3], dim=-1).squeeze(dim=0) + output)
                output = F.softmax(output, dim=-1)
                output_logits.append(output)

                input_token = torch.argmax(output, dim=-1)
                input_token = input_token.item()
                item_tensor = self.dropout(
                    self.embeddings[2](
                        torch.LongTensor([input_token]).unsqueeze(dim=0).to(self.device)))  # (1, seq, codes)

        i3_input_tensor = self.dropout(
            self.embeddings[2](torch.LongTensor(input[2]).unsqueeze(dim=0).to(self.device)))  # (1, seq, codes)
        o3, (ch3, m3, r3) = \
            self.encoders[2](i3_input_tensor, (None, None, None) if i3_state is None else i3_state)
        # save memory state
        i3_state = (ch3, m3, r3)

        return torch.cat(output_logits, dim=0), i1_state, i2_state, i3_state
