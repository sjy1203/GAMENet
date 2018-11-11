import torch
import torch.nn as nn
import numpy as np
from models import CycleRNN, MANN, LSTM_CAT, RNN_Attn, MANN_Attn
from util import metric, transform_split, llprint
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F

model_name = 'MANN_Attn'
resume_name = 'Epoch_4_Loss_2.8756.model'

def eval(model, data_eval, voc_size, epoch):
    eval_len = len(data_eval)
    # evaluate
    print('')
    model.eval()
    y_pred_prob = np.zeros((eval_len, voc_size[-1]))
    y_gt = y_pred_prob.copy()
    y_pred = y_pred_prob.copy()

    for step, input in enumerate(data_eval):
        pre_outputs, pre_labels, last_outputs, last_labels = model(input)
        last_outputs = F.softmax(last_outputs, dim=-1)
        last_v, last_arg = torch.max(last_outputs, dim=-1)
        last_v = last_v.detach().cpu().numpy()
        last_arg = last_arg.detach().cpu().numpy()

        def filter_other_token(x):
            if x[1] >= voc_size[-1]:
                return False
            return True
        try:
            last_v, last_arg = zip(*filter(filter_other_token, zip(last_v, last_arg)))
        except Exception:
            last_v, last_arg = [], []

        last_v, last_arg = list(last_v), list(last_arg)
        target = last_labels.detach().cpu().numpy()[:-1] # remove end token

        pred_prob = np.zeros(voc_size[-1])
        pred_prob[last_arg] = last_v
        pred = pred_prob.copy()
        pred[last_arg] = 1
        y_pred[step, :] = pred
        y_pred_prob[step, :] = pred_prob
        y_gt[step, target] = 1

        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))

    js, auc, p_1, p_3, p_5, f1, auprc = metric(y_gt, y_pred, y_pred_prob)
    llprint('\tJS: %.4f, AUC: %.4f, P1: %.4f, P3: %.4f, P5: %.4f, F1: %.4f, AUPRC: %.4F\n' % (
        js, auc, p_1, p_3, p_5, f1, auprc
    ))


if __name__ == '__main__':
    if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))

    data_path = '../data/records.pkl'
    voc_path = '../data/voc.pkl'
    device = torch.device('cuda:0')

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_eval = data[split_point:split_point + eval_len]

    EPOCH = 30
    LR = 0.001
    EVAL = True

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    model = MANN_Attn(voc_size, device=device)
    if EVAL:
        model.load_state_dict(torch.load(open(os.path.join("saved", model_name, resume_name), 'rb')))
    model.to(device=device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=LR)

    if EVAL:
        eval(model, data_eval, voc_size, 0)
    else:
        for epoch in range(EPOCH):
            loss_record = []
            start_time = time.time()
            model.train()
            for step, input in enumerate(data_train):
                pre_outputs, pre_labels, last_outputs, last_labels = model(input)
                last_loss = criterion(last_outputs, last_labels)
                optimizer.zero_grad()
                last_loss.backward()
                optimizer.step()
                loss_record.append(last_loss.item())

                llprint('\rTrain--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_train)))

            eval(model, data_eval, voc_size, epoch)

            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            llprint('\n\tEpoch: %d, Loss: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                                np.mean(loss_record),
                                                                                                elapsed_time,
                                                                                                elapsed_time * (
                                                                                                            EPOCH - epoch - 1)/60))

            torch.save(model.state_dict(), open( os.path.join('saved', model_name, 'Epoch_%d_Loss_%.4f.model' % (epoch, np.mean(loss_record))), 'wb'))
            print('')

        # test
        torch.save(model.state_dict(), open(
            os.path.join('saved', model_name, 'final.model'), 'wb'))
