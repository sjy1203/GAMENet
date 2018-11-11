import dill
import numpy as np

import sys
sys.path.append("..")
from util import multi_label_metric

data_path = '../../data/records_final.pkl'
voc_path = '../../data/voc_final.pkl'

ddi_adj_path = '../../data/ddi_A_final.pkl'

data = dill.load(open(data_path, 'rb'))
voc = dill.load(open(voc_path, 'rb'))
diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
split_point = int(len(data) * 2 / 3)
data_train = data[:split_point]
eval_len = int(len(data[split_point:]) / 2)
data_test = data[split_point:split_point + eval_len]
data_eval = data[split_point+eval_len:]


def main():
    gt = []
    pred = []
    for patient in data_test:
        if len(patient) == 1:
            continue
        for adm_idx, adm in enumerate(patient):
            if adm_idx < len(patient)-1:
                gt.append(patient[adm_idx+1][2])
                pred.append(adm[2])
    med_voc_size = len(med_voc.idx2word)
    y_gt = np.zeros((len(gt), med_voc_size))
    y_pred = np.zeros((len(gt), med_voc_size))
    for idx, item in enumerate(gt):
        y_gt[idx, item] = 1
    for idx, item in enumerate(pred):
        y_pred[idx, item] = 1

    ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(y_gt, y_pred, y_pred)

    # ddi rate
    ddi_A = dill.load(open(ddi_adj_path, 'rb'))
    all_cnt = 0
    dd_cnt = 0
    med_cnt = 0
    visit_cnt = 0
    for adm in y_pred:
        med_code_set = np.where(adm == 1)[0]
        visit_cnt += 1
        med_cnt += len(med_code_set)
        for i, med_i in enumerate(med_code_set):
            for j, med_j in enumerate(med_code_set):
                if j <= i:
                    continue
                all_cnt += 1
                if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                    dd_cnt += 1
    ddi_rate = dd_cnt / all_cnt
    print('\tDDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1
    ))
    print('avg med', med_cnt/ visit_cnt)

if __name__ == '__main__':
    main()