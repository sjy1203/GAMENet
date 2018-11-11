import torch
import torch.nn as nn
from sklearn.metrics import jaccard_similarity_score, roc_auc_score, precision_score, f1_score, average_precision_score
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
from collections import defaultdict
import torch.nn.functional as F

import sys
sys.path.append("..")
from models import DMNC
from util import llprint, sequence_metric, ddi_rate_score, get_n_params

torch.manual_seed(1203)
model_name = 'DMNC'
resume_name = ''

'''
It's better to refer to the offical implement in tensorflow.  https://github.com/thaihungle/DMNC
'''

def sequence_output_process(output_logits, filter_token):
    pind = np.argsort(output_logits, axis=-1)[:, ::-1]
    out_list = []
    for i in range(len(pind)):
        for j in range(pind.shape[1]):
            label = pind[i][j]
            if label in filter_token:
                continue
            if label not in out_list:
                out_list.append(label)
                break
    y_pred_prob_tmp = []
    for idx, item in enumerate(out_list):
        y_pred_prob_tmp.append(output_logits[idx, item])
    sorted_predict = [x for _, x in sorted(zip(y_pred_prob_tmp, out_list), reverse=True)]
    return out_list, sorted_predict

def eval(model, data_eval, voc_size, epoch):
    # evaluate
    print('')
    model.eval()

    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    records = []
    for step, input in enumerate(data_eval):
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        i1_state, i2_state, i3_state = None, None, None
        for adm in input:
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            output_logits, i1_state, i2_state, i3_state = model(adm, i1_state, i2_state, i3_state)
            output_logits = output_logits.detach().cpu().numpy()

            out_list, sorted_predict = sequence_output_process(output_logits, [voc_size[2], voc_size[2]+1])

            y_pred_label.append(sorted_predict)
            y_pred_prob.append(np.mean(output_logits[:,:-2], axis=0))

            y_pred_tmp = np.zeros(voc_size[2])
            y_pred_tmp[out_list] = 1
            y_pred.append(y_pred_tmp)
        records.append(y_pred_label)

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = sequence_metric(np.array(y_gt), np.array(y_pred),
                                                                              np.array(y_pred_prob),
                                                                              np.array(y_pred_label))
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)

        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(records)
    llprint('\tDDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
    ))
    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)

def main():
    if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))

    data_path = '../data/records_final.pkl'
    voc_path = '../data/voc_final.pkl'
    device = torch.device('cuda:0')

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    EPOCH = 30
    LR = 0.0005
    TEST = False
    END_TOKEN = voc_size[2] + 1

    model = DMNC(voc_size, device=device)
    if TEST:
        model.load_state_dict(torch.load(open(os.path.join("saved", model_name, resume_name), 'rb')))
    model.to(device=device)
    print('parameters', get_n_params(model))

    criterion2 = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=LR)

    if TEST:
        eval(model, data_test, voc_size, 0)
    else:
        history = defaultdict(list)
        for epoch in range(EPOCH):
            loss_record1 = []
            loss_record2 = []
            start_time = time.time()
            model.train()
            for step, input in enumerate(data_train):
                i1_state, i2_state, i3_state = None, None, None
                for adm in input:
                    loss_target = adm[2] + [END_TOKEN]
                    output_logits, i1_state, i2_state, i3_state = model(adm, i1_state, i2_state, i3_state)
                    loss = criterion2(output_logits, torch.LongTensor(loss_target).to(device))

                    loss_record1.append(loss.item())
                    loss_record2.append(loss.item())

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

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
            llprint('\tEpoch: %d, Loss1: %.4f, Loss2: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                                np.mean(loss_record1),
                                                                                                np.mean(loss_record2),
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