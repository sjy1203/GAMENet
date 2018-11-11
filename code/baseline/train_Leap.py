import torch
import torch.nn as nn
from sklearn.metrics import jaccard_similarity_score, roc_auc_score, precision_score, f1_score, average_precision_score
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F
import random
from collections import defaultdict

import sys
sys.path.append("..")
from models import Leap
from util import llprint, sequence_metric, sequence_output_process, ddi_rate_score, get_n_params

torch.manual_seed(1203)

model_name = 'Leap'
resume_name = ''

def eval(model, data_eval, voc_size, epoch):
    # evaluate
    print('')
    model.eval()

    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    records = []
    med_cnt = 0
    visit_cnt = 0
    for step, input in enumerate(data_eval):
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        for adm in input:
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            output_logits = model(adm)
            output_logits = output_logits.detach().cpu().numpy()

            out_list, sorted_predict = sequence_output_process(output_logits, [voc_size[2], voc_size[2]+1])

            y_pred_label.append(sorted(sorted_predict))
            y_pred_prob.append(np.mean(output_logits[:, :-2], axis=0))

            y_pred_tmp = np.zeros(voc_size[2])
            y_pred_tmp[out_list] = 1
            y_pred.append(y_pred_tmp)
            visit_cnt += 1
            med_cnt += len(sorted_predict)
        records.append(y_pred_label)

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = sequence_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob), np.array(y_pred_label))
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
    print('avg med', med_cnt / visit_cnt)
    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)

def main():
    if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))

    data_path = '../../data/records_final.pkl'
    voc_path = '../../data/voc_final.pkl'
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
    LR = 0.0002
    TEST = False
    END_TOKEN = voc_size[2] + 1

    model = Leap(voc_size, device=device)
    if TEST:
        model.load_state_dict(torch.load(open(os.path.join("saved", model_name, resume_name), 'rb')))
        # pass

    model.to(device=device)
    print('parameters', get_n_params(model))

    optimizer = Adam(model.parameters(), lr=LR)

    if TEST:
        eval(model, data_test, voc_size, 0)
    else:
        history = defaultdict(list)
        for epoch in range(EPOCH):
            loss_record = []
            start_time = time.time()
            model.train()
            for step, input in enumerate(data_train):
                for adm in input:
                    loss_target = adm[2] + [END_TOKEN]
                    output_logits = model(adm)
                    loss = F.cross_entropy(output_logits, torch.LongTensor(loss_target).to(device))

                    loss_record.append(loss.item())

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
            llprint('\tEpoch: %d, Loss1: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                                np.mean(loss_record),
                                                                                                elapsed_time,
                                                                                                elapsed_time * (
                                                                                                            EPOCH - epoch - 1)/60))

            torch.save(model.state_dict(), open( os.path.join('saved', model_name, 'Epoch_%d_JA_%.4f_DDI_%.4f.model' % (epoch, ja, ddi_rate)), 'wb'))
            print('')

        dill.dump(history, open(os.path.join('saved', model_name, 'history.pkl'), 'wb'))
        # test
        torch.save(model.state_dict(), open(
            os.path.join('saved', model_name, 'final.model'), 'wb'))

def fine_tune(fine_tune_name=''):
    data_path = '../../data/records_final.pkl'
    voc_path = '../../data/voc_final.pkl'
    device = torch.device('cuda:0')

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    ddi_A = dill.load(open('../../data/ddi_A_final.pkl', 'rb'))

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    # data_eval = data[split_point+eval_len:]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    model = Leap(voc_size, device=device)
    model.load_state_dict(torch.load(open(os.path.join("saved", model_name, fine_tune_name), 'rb')))
    model.to(device)

    EPOCH = 30
    LR = 0.0001
    END_TOKEN = voc_size[2] + 1

    optimizer = Adam(model.parameters(), lr=LR)
    ddi_rate_record = []
    for epoch in range(1):
        loss_record = []
        start_time = time.time()
        random_train_set = [ random.choice(data_train) for i in range(len(data_train))]
        for step, input in enumerate(random_train_set):
            model.train()
            K_flag = False
            for adm in input:
                target = adm[2]
                output_logits = model(adm)
                out_list, sorted_predict = sequence_output_process(output_logits.detach().cpu().numpy(), [voc_size[2], voc_size[2] + 1])

                inter = set(out_list) & set(target)
                union = set(out_list) | set(target)
                jaccard = 0 if union == 0 else len(inter) / len(union)
                K = 0
                for i in out_list:
                    if K == 1:
                        K_flag = True
                        break
                    for j in out_list:
                        if ddi_A[i][j] == 1:
                            K = 1
                            break

                loss = -jaccard * K * torch.mean(F.log_softmax(output_logits, dim=-1))


                loss_record.append(loss.item())

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            llprint('\rTrain--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_train)))

            if K_flag:
                ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_test, voc_size, epoch)


                end_time = time.time()
                elapsed_time = (end_time - start_time) / 60
                llprint('\tEpoch: %d, Loss1: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                               np.mean(loss_record),
                                                                                               elapsed_time,
                                                                                               elapsed_time * (
                                                                                                       EPOCH - epoch - 1) / 60))

                torch.save(model.state_dict(),
                   open(os.path.join('saved', model_name, 'fine_Epoch_%d_JA_%.4f_DDI_%.4f.model' % (epoch, ja, ddi_rate)),
                        'wb'))
                print('')

    # test
    torch.save(model.state_dict(), open(
        os.path.join('saved', model_name, 'final.model'), 'wb'))



if __name__ == '__main__':
    # main()
    fine_tune(fine_tune_name='Epoch_26_JA_0.4465_DDI_0.0723.model')