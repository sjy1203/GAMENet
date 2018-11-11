import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
import random
import math
import numpy as np
import dill
from util import llprint, sequence_metric, sequence_output_process, ddi_rate_score
import os
from collections import defaultdict

MAX_LEN = 20
model_name = 'seq2seq_small'
if not os.path.exists(os.path.join("saved", model_name)):
    os.makedirs(os.path.join("saved", model_name))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1

data_path = '../data/records_final.pkl'
voc_path = '../data/voc_final.pkl'

data = dill.load(open(data_path, 'rb'))
voc = dill.load(open(voc_path, 'rb'))
diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

pairs = []

for patient in data:
    for visit in patient:
        i1 = visit[0]
        # i2 = visit[1]
        o = visit[2]

        input_seq = list(np.array(i1) + 2)
        # input_seq.extend(list(np.array(i2) + 2 + len(diag_voc.idx2word)))
        input_seq.append(EOS_token)
        if len(input_seq) > MAX_LEN:
            MAX_LEN = len(input_seq)
        output_seq = list(np.array(o)+2)
        output_seq.append(EOS_token)

        pairs.append((input_seq, output_seq))
print('MAX_LEN', MAX_LEN)
split_point = int(len(pairs) * 2 / 3)
train_pairs = pairs[:split_point]
eval_len = int(len(pairs[split_point:]) / 2)
test_pairs = pairs[split_point:split_point + eval_len]
eval_pairs = pairs[split_point+eval_len:]


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LEN):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = self.out(output[0])
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(MAX_LEN, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def tensorsFromPair(pair):
    return (torch.tensor(pair[0], dtype=torch.long, device=device).view(-1, 1),
            torch.tensor(pair[1], dtype=torch.long, device=device).view(-1, 1))


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = [tensorsFromPair(random.choice(train_pairs))
                      for i in range(n_iters)]
    criterion = nn.CrossEntropyLoss()
    history = defaultdict(list)
    for epoch in range(30):
        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = train(input_tensor, target_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            llprint('\rTrain--Epoch: %d, Step: %d/%d' % (epoch, iter, n_iters))

        print_loss_avg = print_loss_total / n_iters
        print_loss_total = 0

        #eval
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        for pair in eval_pairs:
            y_gt_tmp = np.zeros(len(med_voc.idx2word))
            y_gt_tmp[np.array(pair[1])[:-1]-2] = 1
            y_gt.append(y_gt_tmp)

            input_tensor, output_tensor = tensorsFromPair(pair)
            output_logits = evaluate(encoder, decoder, input_tensor)
            output_logits = F.softmax(output_logits)
            output_logits = output_logits.detach().cpu().numpy()
            out_list, sorted_predict = sequence_output_process(output_logits, [SOS_token, EOS_token])

            y_pred_label.append(np.array(sorted_predict)-2)
            y_pred_prob.append(np.mean(output_logits[:, 2:], axis=0))

            y_pred_tmp = np.zeros(len(med_voc.idx2word))
            if len(out_list) != 0 :
                y_pred_tmp[np.array(out_list) - 2] = 1
            y_pred.append(y_pred_tmp)

        ja, prauc, avg_p, avg_r, avg_f1 = sequence_metric(np.array(y_gt), np.array(y_pred),
                                                        np.array(y_pred_prob),
                                                        np.array(y_pred_label))
        # ddi rate
        ddi_A = dill.load(open('../data/ddi_A_final.pkl', 'rb'))
        all_cnt = 0
        dd_cnt = 0
        for adm in y_pred_label:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        dd_cnt += 1
        ddi_rate = dd_cnt / all_cnt

        history['ja'].append(ja)
        history['ddi_rate'].append(ddi_rate)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)
        llprint('\n\tDDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1
        ))

        dill.dump(history, open(os.path.join('saved', model_name, 'history.pkl'), 'wb'))

        torch.save(encoder.state_dict(),
                   open(
                       os.path.join('saved', model_name, 'encoder_Epoch_%d_JA_%.4f_DDI_%.4f.model' % (epoch, ja, dd_cnt/all_cnt)),
                       'wb'))
        torch.save(decoder.state_dict(),
                   open(
                       os.path.join('saved', model_name, 'decoder_Epoch_%d_JA_%.4f_DDI_%.4f.model' % (epoch, ja, dd_cnt/all_cnt)),
                       'wb'))




def evaluate(encoder, decoder, input_tensor, max_length=MAX_LEN):
    with torch.no_grad():
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        output_logits = []

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)

            output_logits.append(decoder_output)

            decoder_input = topi.squeeze().detach()

        return torch.cat(output_logits, dim=0)

hidden_size = 64
is_train = True
encoder1 = EncoderRNN(2 + len(diag_voc.idx2word) + len(pro_voc.idx2word), hidden_size)
decoder1 = AttnDecoderRNN(hidden_size, len(med_voc.idx2word)+2, dropout_p=0.1)


if is_train:
    # training
    encoder1.to(device)
    decoder1.to(device)
    trainIters(encoder1, decoder1, len(train_pairs))
else:
    # eval
    encoder1.load_state_dict(torch.load(open(os.path.join("saved", model_name, 'encoder_Epoch_10_P@3_0.3293_PRAUC_0.3246.model'), 'rb')))
    decoder1.load_state_dict(torch.load(open(os.path.join("saved", model_name, 'decoder_Epoch_10_p_3_0.3293_PRAUC_0.3246.model'), 'rb')))
    encoder1.to(device)
    decoder1.to(device)
    # y_pred_label = []
    # for pair in eval_pairs:
    #     input_tensor, output_tensor = tensorsFromPair(pair)
    #     output_logits = evaluate(encoder1, decoder1, input_tensor)
    #     output_logits = F.softmax(output_logits)
    #     output_logits = output_logits.detach().cpu().numpy()
    #     out_list, sorted_predict = sequence_output_process(output_logits, [SOS_token, EOS_token])
    #     y_pred_label.append(np.array(sorted_predict) - 2)

    # test
    y_gt = []
    y_pred = []
    y_pred_prob = []
    y_pred_label = []
    sample = False
    if sample:
        # sample_data = [[[[124, 19, 41, 87, 172], [146, 98], [10, 9]],
        #                 [[205, 38, 75, 276, 48, 19, 123, 59, 106, 134, 68, 381, 3, 182],
        #                  [346, 59, 147, 146, 63, 96, 60, 39, 98, 2, 48],
        #                  [13, 18, 41, 8, 0, 21, 56, 42, 10, 16]]]]
        diabetes_data = dill.load(open('../data/diabetes_data.pkl', 'rb'))
        heart_data = dill.load(open('../data/heart_data.pkl', 'rb'))
        hypertension_data = dill.load(open('../data/hypertension_data.pkl', 'rb'))
        Neurotic_data = dill.load(open('../data/Neurotic_data.pkl', 'rb'))
        Lymphoid_data = dill.load(open('../data/Lymphoid_data.pkl', 'rb'))

        sample_data = hypertension_data
        test_pairs = []
        for patient in sample_data:
            for visit in patient:
                i1 = visit[0]
                i2 = visit[1]
                o = visit[2]

                input_seq = list(np.array(i1) + 2)
                input_seq.extend(list(np.array(i2) + 2 + len(diag_voc.idx2word)))
                input_seq.append(EOS_token)
                if len(input_seq) > MAX_LEN:
                    MAX_LEN = len(input_seq)
                output_seq = list(np.array(o) + 2)
                output_seq.append(EOS_token)

                test_pairs.append((input_seq, output_seq))

    for pair in test_pairs:
        y_gt_tmp = np.zeros(len(med_voc.idx2word))
        y_gt_tmp[np.array(pair[1])[:-1] - 2] = 1
        y_gt.append(y_gt_tmp)

        input_tensor, output_tensor = tensorsFromPair(pair)
        output_logits = evaluate(encoder1, decoder1, input_tensor)
        output_logits = F.softmax(output_logits)
        output_logits = output_logits.detach().cpu().numpy()
        out_list, sorted_predict = sequence_output_process(output_logits, [SOS_token, EOS_token])

        y_pred_label.append(np.array(sorted_predict) - 2)
        y_pred_prob.append(np.mean(output_logits[:, 2:], axis=0))

        y_pred_tmp = np.zeros(len(med_voc.idx2word))
        if len(out_list) != 0:
            y_pred_tmp[np.array(out_list) - 2] = 1
        y_pred.append(y_pred_tmp)

    ja, prauc, avg_p, avg_r, avg_f1 = sequence_metric(np.array(y_gt), np.array(y_pred),
                                                    np.array(y_pred_prob),
                                                    np.array(y_pred_label))
    # ddi rate
    ddi_A = dill.load(open('../data/ddi_A_final.pkl', 'rb'))
    all_cnt = 0
    dd_cnt = 0
    for adm in y_pred_label:
        med_code_set = adm
        for i, med_i in enumerate(med_code_set):
            for j, med_j in enumerate(med_code_set):
                if j <= i:
                    continue
                all_cnt += 1
                if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                    dd_cnt += 1
    ddi_rate = dd_cnt / all_cnt
    llprint('\n\tDDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1
    ))







