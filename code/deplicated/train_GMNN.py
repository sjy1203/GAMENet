import torch
import torch.nn as nn
from sklearn.metrics import jaccard_similarity_score, roc_auc_score, precision_score, f1_score, average_precision_score, average_precision_score
import numpy as np
from models import GMNN
from util import llprint, multi_label_metric
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F
torch.manual_seed(1203)
model_name = 'GMNN_1'
resume_name = 'Epoch_4_Loss1_1.3970.model'

def eval(model, data_eval, voc_size, epoch):
    # evaluate
    print('')
    model.eval()
    smm_record = []
    auc, p_1, p_3, p_5, f1, prauc = [[] for _ in range(6)]
    for step, input in enumerate(data_eval):
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        input1_hidden, input2_hidden, target_hidden= None, None, None
        for adm in input:
            y_pred_label_tmp = []
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            target_output1, output_logits, output_labels, [input1_hidden, input2_hidden, target_hidden] = model(adm, [input1_hidden, input2_hidden, target_hidden])

            target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
            a = np.argsort(target_output1)[::-1]
            b = np.max(output_logits, axis=-1)
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
        adm_auc, adm_p_1, adm_p_3, adm_p_5, adm_f1, adm_prauc = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        auc.append(adm_auc)
        p_1.append(adm_p_1)
        p_3.append(adm_p_3)
        p_5.append(adm_p_5)
        f1.append(adm_f1)
        prauc.append(adm_prauc)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))

    llprint('\tAUC: %.4f, P1: %.4f, P3: %.4f, P5: %.4f, F1: %.4f, PRAUC: %.4f\n' % (
        np.mean(auc), np.mean(p_1), np.mean(p_3), np.mean(p_5), np.mean(f1), np.mean(prauc)
    ))
    dill.dump(obj=smm_record, file=open('../data/smm_records.pkl', 'wb'))



def main():
    if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))

    data_path = '../data/records.pkl'
    voc_path = '../data/voc.pkl'
    ehr_adj_path = '../data/ehr_adj.pkl'
    ddi_adj_path = '../data/ddi_A.pkl'
    device = torch.device('cuda:0')

    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    # data_eval = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]

    EPOCH = 30
    LR = 0.001
    EVAL = True

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    model = GMNN(voc_size, ehr_adj, ddi_adj, emb_dim=64, device=device)
    if EVAL:
        model.load_state_dict(torch.load(open(os.path.join("saved", model_name, resume_name), 'rb')))
    model.to(device=device)

    optimizer = Adam(list(model.parameters()), lr=LR)

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
                loss = 0
                for adm in input:
                    loss1_target = np.zeros((1, voc_size[2]))
                    loss1_target[:, adm[2]] = 1

                    loss2_target = adm[2] + [adm[2][0]]

                    loss3_target = np.full((1, voc_size[2]), -1)
                    for idx, item in enumerate(adm[2]):
                        loss3_target[0][idx] = item

                    target_output1, target_output2, [input1_hidden, input2_hidden, target_hidden], batch_pos_loss, batch_neg_loss = model(adm, [input1_hidden, input2_hidden, target_hidden])

                    loss1 = F.binary_cross_entropy_with_logits(target_output1, torch.FloatTensor(loss1_target).to(device))
                    loss2 = F.cross_entropy(target_output2, torch.LongTensor(loss2_target).to(device))

                    # loss = 9*loss1/10 + loss2/10
                    loss3 = F.multilabel_margin_loss(F.sigmoid(target_output1), torch.LongTensor(loss3_target).to(device))
                    loss += loss1 + 0.1*loss3 + 0.01*batch_neg_loss

                    loss_record1.append(loss.item())
                    loss_record2.append(loss3.item())

                optimizer.zero_grad()
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