import torch
import torch.nn as nn
from sklearn.metrics import jaccard_similarity_score, roc_auc_score, precision_score, f1_score, average_precision_score
import numpy as np
from models import RNN_Two
from util import llprint
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F

model_name = 'RNN_Two'
resume_name = 'Epoch_4_Loss_2.8756.model'

def f1_dnc(y_gt, y_pred):
    all_micro = []
    for b in range(y_gt.shape[0]):
        all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
    return np.mean(all_micro)

def roc_auc_dnc(y_gt, y_prob):
    all_micro = []
    for b in range(len(y_gt)):
        all_micro.append(roc_auc_score(y_gt[b], y_prob[b], average='micro'))
    return np.mean(all_micro)

def precision_at_k_v2(y_gt, y_prob, k=3):
    precision = 0
    sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
    for i in range(len(y_gt)):
        TP = 0
        for j in range(len(sort_index[i])):
            if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
        precision += TP / len(sort_index[i])
    return precision / len(y_gt)


def metric(y_eval, y_pred, y_pred_prob):

    auc = roc_auc_dnc(y_eval, y_pred_prob)
    p_1 = precision_at_k_v2(y_eval, y_pred_prob, k=1)
    p_3 = precision_at_k_v2(y_eval, y_pred_prob, k=3)
    p_5 = precision_at_k_v2(y_eval, y_pred_prob, k=5)
    f1 = f1_dnc(y_eval, y_pred)

    return auc, p_1, p_3, p_5, f1

def eval(model, data_eval, voc_size, epoch):
    # evaluate
    print('')
    model.eval()

    auc, p_1, p_3, p_5, f1 = [[] for _ in range(5)]
    for step, input in enumerate(data_eval):
        y_gt = []
        y_pred = []
        y_pred_prob = []
        input1_hidden, input2_hidden, target_hidden = None, None, None
        for adm in input:
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            output2_logits, output2_labels = model.seq_evaluate(target_hidden, max_len=15)
            target_output1, _, [input1_hidden, input2_hidden, target_hidden] = model(adm, [input1_hidden, input2_hidden, target_hidden])

            target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0][0]
            y_pred_prob.append(target_output1)
            y_pred_tmp = target_output1.copy()
            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0
            y_pred.append(y_pred_tmp)
            print('adm')
        adm_auc, adm_p_1, adm_p_3, adm_p_5, adm_f1 = metric(y_gt, y_pred, y_pred_prob)
        auc.append(adm_auc)
        p_1.append(adm_p_1)
        p_3.append(adm_p_3)
        p_5.append(adm_p_5)
        f1.append(adm_f1)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))

    llprint('\tAUC: %.4f, P1: %.4f, P3: %.4f, P5: %.4f, F1: %.4f\n' % (
        np.mean(auc), np.mean(p_1), np.mean(p_3), np.mean(p_5), np.mean(f1)
    ))


        # last_outputs = F.softmax(last_outputs, dim=-1)
        # last_v, last_arg = torch.max(last_outputs, dim=-1)
        # last_v = last_v.detach().cpu().numpy()
        # last_arg = last_arg.detach().cpu().numpy()
        #
        # def filter_other_token(x):
        #     if x[1] >= voc_size[-1]:
        #         return False
        #     return True
        # try:
        #     last_v, last_arg = zip(*filter(filter_other_token, zip(last_v, last_arg)))
        # except Exception:
        #     last_v, last_arg = [], []
        #
        # last_v, last_arg = list(last_v), list(last_arg)
        # target = last_labels.detach().cpu().numpy()[:-1] # remove end token
        #
        # pred_prob = np.zeros(voc_size[-1])
        # pred_prob[last_arg] = last_v
        # pred = pred_prob.copy()
        # pred[last_arg] = 1
        # y_pred[step, :] = pred
        # y_pred_prob[step, :] = pred_prob
        # y_gt[step, target] = 1






def main():
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
    model = RNN_Two(voc_size, device=device)
    if EVAL:
        pass
        #model.load_state_dict(torch.load(open(os.path.join("saved", model_name, resume_name), 'rb')))
    model.to(device=device)

    criterion1 = nn.BCEWithLogitsLoss().to(device)
    criterion2 = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=LR)

    if EVAL:
        eval(model, data_eval, voc_size, 0)
    else:
        for epoch in range(EPOCH):
            loss_record1 = []
            loss_record2 = []
            start_time = time.time()
            model.train()
            for step, input in enumerate(data_train):
                input1_hidden, input2_hidden, target_hidden = None, None, None
                for adm in input:
                    loss1_target = np.zeros((1, voc_size[2]))
                    loss1_target[:, adm[2]] = 1
                    loss2_target = adm[2] + [voc_size[2]+1]

                    target_output1, target_output2, [input1_hidden, input2_hidden, target_hidden] = model(adm, [input1_hidden, input2_hidden, target_hidden])

                    loss1 = criterion1(target_output1, torch.LongTensor(loss1_target).to(device))
                    loss2 = criterion2(target_output2, torch.LongTensor(loss2_target).to(device))

                    optimizer.zero_grad()
                    loss1.backward()
                    loss2.backward()
                    optimizer.step()
                    loss_record1.append(loss1.item())
                    loss_record2.append(loss2.item())

                llprint('\rTrain--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_train)))

            eval(model, data_eval, voc_size, epoch)

            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            llprint('\tEpoch: %d, Loss1: %.4f, Loss2: %.4f One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                                np.mean(loss_record1),
                                                                                                np.mean(loss_record2),
                                                                                                elapsed_time,
                                                                                                elapsed_time * (
                                                                                                            EPOCH - epoch - 1)/60))

            torch.save(model.state_dict(), open( os.path.join('saved', model_name, 'Epoch_%d_Loss1_%.4f.model' % (epoch, np.mean(loss_record1))), 'wb'))
            print('')

        # test
        torch.save(model.state_dict(), open(
            os.path.join('saved', model_name, 'final.model'), 'wb'))


if __name__ == '__main__':
    main()