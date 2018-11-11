import torch
import torch.nn as nn
from sklearn.metrics import jaccard_similarity_score, roc_auc_score, precision_score, f1_score, average_precision_score, average_precision_score
import numpy as np
from models import GAMENet
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F
from collections import defaultdict

import sys
sys.path.append('..')
from util import llprint, multi_label_metric, ddi_rate_score

torch.manual_seed(1203)
model_name = 'GAMENet'
resume_name = ''


def eval(model, data_eval, voc_size, epoch):
    # evaluate
    print('')
    model.eval()
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    for step, input in enumerate(data_eval):
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        input1_hidden, input2_hidden, target_hidden = None, None, None
        prev_target = None
        for adm_idx, adm in enumerate(input):

            target_output1, [input1_hidden, input2_hidden, target_hidden] = model(adm, prev_target, [input1_hidden, input2_hidden, target_hidden])
            prev_target = adm[2]

            y_pred_label_tmp = []
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output1)
            y_pred_tmp = target_output1.copy()
            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0
            y_pred.append(y_pred_tmp)
            for idx, value in enumerate(y_pred_tmp):
                if value == 1:
                    y_pred_label_tmp.append(idx)
            y_pred_label.append(y_pred_label_tmp)
        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))
    # ddi rate
    ddi_rate = ddi_rate_score(smm_record)

    llprint('\tDDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
    ))

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


def main():
    if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))

    data_path = '../data/records_final.pkl'
    voc_path = '../data/voc_final.pkl'

    ehr_adj_path = '../data/ehr_adj_final.pkl'
    ddi_adj_path = '../data/ddi_A_final.pkl'
    device = torch.device('cuda:0')


    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]

    EPOCH = 30
    LR = 0.001
    TEST = True
    Neg_Loss = False
    TARGET_DDI = 0.001

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    model = GAMENet(voc_size, ehr_adj, ddi_adj, emb_dim=64, device=device, with_memory=True)
    if TEST:
        model.load_state_dict(torch.load(open(os.path.join("saved", model_name, resume_name), 'rb')))
    model.to(device=device)

    optimizer = Adam(list(model.parameters()), lr=LR)

    if TEST:
        eval(model, data_test, voc_size, 0)
    else:
        history = defaultdict(list)
        for epoch in range(EPOCH):
            loss_record1 = []
            start_time = time.time()
            model.train()
            for step, input in enumerate(data_train):
                input1_hidden, input2_hidden, target_hidden = None, None, None
                loss = 0
                prev_target = None
                for adm in input:
                    loss1_target = np.zeros((1, voc_size[2]))
                    loss1_target[:, adm[2]] = 1

                    loss3_target = np.full((1, voc_size[2]), -1)
                    for idx, item in enumerate(adm[2]):
                        loss3_target[0][idx] = item

                    target_output1, [input1_hidden, input2_hidden, target_hidden], batch_neg_loss = model(adm, prev_target, [input1_hidden, input2_hidden, target_hidden])
                    prev_target = adm[2]

                    loss1 = F.binary_cross_entropy_with_logits(target_output1, torch.FloatTensor(loss1_target).to(device))

                    # loss = 9*loss1/10 + loss2/10
                    loss3 = F.multilabel_margin_loss(F.sigmoid(target_output1), torch.LongTensor(loss3_target).to(device))
                    # loss += loss1 + 0.1*loss3 + 0.01*batch_neg_loss
                    if Neg_Loss:
                        # neg_loss_weight = 0.0007 * (2 ** (epoch // 5))
                        # if neg_loss_weight > 0.01:
                        #     # decay stop:
                        #     neg_loss_weight = 0.01
                        # loss += 0.9*loss1 + 0.02*loss3 + neg_loss_weight*batch_neg_loss
                        loss = 0.001 * batch_neg_loss
                    else:
                        loss = 0.9*loss1 + 0.03*loss3

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    loss_record1.append(loss.item())

                llprint('\rTrain--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_train)))

            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_eval, voc_size, epoch)
            history['ja'].append(ja)
            history['ddi_rate'].append(ddi_rate)
            history['avg_p'].append(avg_p)
            history['avg_r'].append(avg_r)
            history['avg_f1'].append(avg_f1)
            history['prauc'].append(prauc)

            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            llprint('\tEpoch: %d, Loss: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                                np.mean(loss_record1),
                                                                                                elapsed_time,
                                                                                                elapsed_time * (
                                                                                                            EPOCH - epoch - 1)/60))

            torch.save(model.state_dict(), open( os.path.join('saved', model_name, 'Epoch_%d_JA_%.4f_DDI_%.4f.model' % (epoch, ja, ddi_rate)), 'wb'))
            print('')

        dill.dump(history, open(os.path.join('saved', model_name, 'history.pkl'), 'wb'))

        # test
        torch.save(model.state_dict(), open(
            os.path.join('saved', model_name, 'final.model'), 'wb'))


if __name__ == '__main__':
    main()