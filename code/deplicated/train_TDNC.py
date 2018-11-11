import torch
import torch.nn as nn
from sklearn.metrics import jaccard_similarity_score, roc_auc_score, precision_score, f1_score, average_precision_score
import numpy as np
from models import MA_DNC
from util import llprint
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F
torch.manual_seed(1203)
model_name = 'MA_DNC'
resume_name = 'Epoch_2_Loss1_6.7981.model'

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
        i1_state, i2_state, i3_state = None, None, None
        for adm in input:
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            output_logits, i1_state, i2_state, i3_state = model(adm, i1_state, i2_state, i3_state)
            a = torch.argmax(output_logits, dim=-1)
            y_pred_prob_tmp = torch.mean(output_logits, dim=0).detach().cpu().numpy()[:-2]
            y_pred_prob.append(y_pred_prob_tmp) # remove start and end token

            # target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
            # a = np.argsort(target_output1)[::-1]
            # b = np.max(output_logits,axis=-1)
            # y_pred_prob.append(target_output1)
            y_pred_tmp = y_pred_prob_tmp.copy()
            y_pred_tmp[y_pred_tmp>=0.3] = 1
            y_pred_tmp[y_pred_tmp<0.3] = 0
            y_pred.append(y_pred_tmp)

        adm_auc, adm_p_1, adm_p_3, adm_p_5, adm_f1 = metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        auc.append(adm_auc)
        p_1.append(adm_p_1)
        p_3.append(adm_p_3)
        p_5.append(adm_p_5)
        f1.append(adm_f1)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))

    llprint('\tAUC: %.4f, P1: %.4f, P3: %.4f, P5: %.4f, F1: %.4f\n' % (
        np.mean(auc), np.mean(p_1), np.mean(p_3), np.mean(p_5), np.mean(f1)
    ))


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
    # data_eval = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))


    EPOCH = 30
    LR = 0.001
    EVAL = True
    END_TOKEN = voc_size[2] + 1

    model = MA_DNC(voc_size, device=device)
    if EVAL:
        pass
        #model.load_state_dict(torch.load(open(os.path.join("saved", model_name, resume_name), 'rb')))
    model.to(device=device)

    criterion1 = nn.BCEWithLogitsLoss().to(device)
    criterion2 = nn.CrossEntropyLoss().to(device)
    criterion3 = nn.MultiLabelMarginLoss().to(device)
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
                i1_state, i2_state, i3_state = None, None, None
                loss = 0
                for adm in input:
                    loss_target = adm[2] + [END_TOKEN]
                    output_logits, i1_state, i2_state, i3_state = model(adm, i1_state, i2_state, i3_state)
                    loss += criterion2(output_logits, torch.LongTensor(loss_target).to(device))

                    # loss1_target = np.zeros((1, voc_size[2]))
                    # loss1_target[:, adm[2]] = 1
                    #
                    # loss2_target = adm[2] + [adm[2][0]]
                    #
                    # loss3_target = np.full((1, voc_size[2]), -1)
                    # for idx, item in enumerate(adm[2]):
                    #     loss3_target[0][idx] = item
                    #
                    # target_output1, target_output2, [input1_hidden, input2_hidden, target_hidden] = model(adm, [input1_hidden, input2_hidden, target_hidden])
                    #
                    #
                    #
                    # loss1 = criterion1(target_output1, torch.FloatTensor(loss1_target).to(device))
                    # loss2 = criterion2(target_output2, torch.LongTensor(loss2_target).to(device))
                    #
                    # # loss = 9*loss1/10 + loss2/10
                    # loss3 = criterion3(F.sigmoid(target_output1), torch.LongTensor(loss3_target).to(device))
                    # loss += 0.85*loss1 + 0.1*loss2 + 0.05*loss3


                    loss_record1.append(loss.item())
                    loss_record2.append(loss.item())

                optimizer.zero_grad()
                loss = loss / len(input)
                loss.backward()
                optimizer.step()

                llprint('\rTrain--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_train)))

            eval(model, data_eval, voc_size, epoch)

            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            llprint('\tEpoch: %d, Loss1: %.4f, Loss2: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
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