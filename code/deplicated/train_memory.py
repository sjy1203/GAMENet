import torch
import torch.nn as nn
import numpy as np
from models import CycleRNN, MANN, LSTM_CAT
from util import metric, transform_split, llprint
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F

model_name = 'lstm_cat'
resume_name = 'Epoch_2_Loss_0.1138.model'

def eval(model, data_eval, voc_size, epoch):
    eval_len = len(data_eval)
    # evaluate
    print('')
    model.eval()
    y_pred_prob = np.zeros((eval_len, voc_size[-1]))
    y_gt = y_pred_prob.copy()
    y_pred = y_pred_prob.copy()

    for step, input in enumerate(data_eval):
        pred_prob, target = model(input)
        pred_prob = F.sigmoid(pred_prob[0]).detach().cpu().numpy()
        target = target[0].detach().cpu().numpy()

        pred = pred_prob.copy()
        pred[pred >= 0.35] = 1
        pred[pred < 0.35] = 0
        y_pred[step, :] = pred
        y_pred_prob[step, :] = pred_prob
        y_gt[step, :] = target

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

    EPOCH = 100
    LR = 0.001
    EVAL = True

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    model = LSTM_CAT(voc_size, device=device)
    if EVAL:
        model.load_state_dict(torch.load(open(os.path.join("saved", model_name, resume_name), 'rb')))
    model.to(device=device)

    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = Adam(model.parameters(), lr=LR)

    if EVAL:
        eval(model, data_eval, voc_size, 0)
    else:
        for epoch in range(EPOCH):
            loss_record = []
            start_time = time.time()
            model.train()
            for step, input in enumerate(data_train):
                output, y_gt = model(input)
                loss = criterion(output, y_gt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_record.append(loss.item())

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
