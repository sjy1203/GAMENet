import dill
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import jaccard_similarity_score
import os

import sys
sys.path.append('..')
from util import multi_label_metric

np.random.seed(1203)
model_name = 'LR'

if not os.path.exists(os.path.join("saved", model_name)):
    os.makedirs(os.path.join("saved", model_name))

def create_dataset(data, diag_voc, pro_voc, med_voc):
    i1_len = len(diag_voc.idx2word)
    i2_len = len(pro_voc.idx2word)
    output_len = len(med_voc.idx2word)
    input_len = i1_len + i2_len
    X = []
    y = []
    for patient in data:
        for visit in patient:
            i1 = visit[0]
            i2 = visit[1]
            o = visit[2]

            multi_hot_input = np.zeros(input_len)
            multi_hot_input[i1] = 1
            multi_hot_input[np.array(i2) + i1_len] = 1

            multi_hot_output = np.zeros(output_len)
            multi_hot_output[o] = 1

            X.append(multi_hot_input)
            y.append(multi_hot_output)

    return np.array(X), np.array(y)


def main():
    grid_search = False
    data_path = '../../data/records_final.pkl'
    voc_path = '../../data/voc_final.pkl'

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_eval = data[split_point+eval_len:]
    data_test = data[split_point:split_point + eval_len]

    train_X, train_y = create_dataset(data_train, diag_voc, pro_voc, med_voc)
    test_X, test_y = create_dataset(data_test, diag_voc, pro_voc, med_voc)
    eval_X, eval_y = create_dataset(data_eval, diag_voc, pro_voc, med_voc)

    if grid_search:
        params = {
            'estimator__penalty': ['l2'],
            'estimator__C': np.linspace(0.00002, 1, 100)
        }

        model = LogisticRegression()
        classifier = OneVsRestClassifier(model)
        lr_gs = GridSearchCV(classifier, params, verbose=1).fit(train_X, train_y)

        print("Best Params", lr_gs.best_params_)
        print("Best Score", lr_gs.best_score_)

        return


    # sample_X, sample_y = create_dataset(sample_data, diag_voc, pro_voc, med_voc)

    model = LogisticRegression(C=0.90909)
    classifier = OneVsRestClassifier(model)
    classifier.fit(train_X, train_y)

    y_pred = classifier.predict(test_X)
    y_prob = classifier.predict_proba(test_X)

    ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(test_y, y_pred, y_prob)

    # ddi rate
    ddi_A = dill.load(open('../../data/ddi_A_final.pkl', 'rb'))
    all_cnt = 0
    dd_cnt = 0
    med_cnt = 0
    visit_cnt = 0
    for adm in y_pred:
        med_code_set = np.where(adm==1)[0]
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
    print('\tDDI Rate: %.4f, Jaccard: %.4f, PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1
    ))

    history = defaultdict(list)
    for i in range(30):
        history['jaccard'].append(ja)
        history['ddi_rate'].append(ddi_rate)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)

    dill.dump(history, open(os.path.join('saved', model_name, 'history.pkl'), 'wb'))

    print('avg med', med_cnt / visit_cnt)


if __name__ == '__main__':
    main()