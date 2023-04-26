import numpy as np
import os


def calculate_metrics(y_true, y_pred):
    from sklearn.metrics import roc_auc_score, average_precision_score
    print(' ===========  test ===========')
    y_pred = np.round(y_pred).astype(np.int8)
    auc_test = roc_auc_score(y_true, y_pred)
    pr_test = average_precision_score(y_true, y_pred)
    tp, fp, tn, fn, accuracy, precision, sensitivity, recall, specificity, MCC, f1_score = calculate_performance(
        len(y_pred), y_pred, y_true)
    return tp, fp, tn, fn, accuracy, precision, sensitivity, recall, specificity, MCC, f1_score, auc_test, pr_test

def calculate_performance(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    accuracy = round(float(tp + tn) / test_num, 4)
    precision = round(float(tp) / (tp + fp + 1e-06), 4)
    sensitivity = round(float(tp) / (tp + fn + 1e-06), 4)
    recall = round(float(tp) / (tp + fn + 1e-06), 4)
    specificity = round(float(tn) / (tn + fp + 1e-06), 4)
    f1_score = round(float(2 * tp) / (2 * tp + fp + fn + 1e-06), 4)
    MCC = round(float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))), 4)
    return tp, fp, tn, fn, accuracy, precision, sensitivity, recall, specificity, MCC, f1_score


partition = True
rewired = True
algorithm = 'topsyturvy'
if partition:
    result_file = open(f"results_{algorithm}/partitions/all_results.tsv", "w")
    result_file.write("Model\tDataset\tMetric\tValue\tSplit\n")
    for dataset in ["du","guo","huang", "richoux", "pan"]:
        print(f"########## Dataset: {dataset} ##########")
        for train in ["both", "0"]:
            for test in ["0", "1"]:
                if train == "0" and test == "0" or not os.path.isfile(f"results_{algorithm}/partitions/{dataset}_{train}_{test}.txt.predictions.tsv"):
                    continue
                y_pred = []
                y_true = []
                with open(f"results_{algorithm}/partitions/{dataset}_{train}_{test}.txt.predictions.tsv", "r") as f:
                    for line in f:
                        y_pred.append(float(line.strip().split("\t")[3]))
                        y_true.append(float(line.strip().split("\t")[2]))
                print(f"******* Train: {train}, test: {test} *******")
                split = f'{train}->{test}'
                tp, fp, tn, fn, accuracy, precision, sensitivity, recall, specificity, MCC, f1_score, auc, pr = calculate_metrics(
                    y_true, y_pred)
                result_file.write(f'{algorithm}\t{dataset}\tAccuracy\t{accuracy}\t{split}\n')
                result_file.write(f'{algorithm}\t{dataset}\tPrecision\t{precision}\t{split}\n')
                result_file.write(f'{algorithm}\t{dataset}\tSensitivity\t{sensitivity}\t{split}\n')
                result_file.write(f'{algorithm}\t{dataset}\tRecall\t{recall}\t{split}\n')
                result_file.write(f'{algorithm}\t{dataset}\tSpecificity\t{specificity}\t{split}\n')
                result_file.write(f'{algorithm}\t{dataset}\tMCC\t{MCC}\t{split}\n')
                result_file.write(f'{algorithm}\t{dataset}\tF1\t{f1_score}\t{split}\n')
                result_file.write(f'{algorithm}\t{dataset}\tAUC\t{auc}\t{split}\n')
                result_file.write(f'{algorithm}\t{dataset}\tAUPR\t{pr}\t{split}\n')
    result_file.close()
else:
    if rewired:
        folder = f'results_{algorithm}/rewired'
        datasets = ["du", "guo", "huang", "pan", "richoux_regular", "richoux_strict"]
        split = 'Rewired'
    else:
        folder = f'results_{algorithm}/original'
        datasets = ["du", "guo", "huang", "pan", "richoux_regular", "richoux_strict", "gold_standard"]
        split = 'Original'
    result_file = open(f"{folder}/all_results.tsv", "w")
    result_file.write("Model\tDataset\tMetric\tValue\tSplit\n")
    for dataset in datasets:
        print(f"########## Dataset: {dataset} ##########")
        y_pred = []
        y_true = []
        with open(f"{folder}/{dataset}.txt.predictions.tsv", "r") as f:
            for line in f:
                y_pred.append(float(line.strip().split("\t")[3]))
                y_true.append(float(line.strip().split("\t")[2]))
        tp, fp, tn, fn, accuracy, precision, sensitivity, recall, specificity, MCC, f1_score, auc, pr = calculate_metrics(y_true, y_pred)
        result_file.write(f'{algorithm}\t{dataset}\tAccuracy\t{accuracy}\t{split}\n')
        result_file.write(f'{algorithm}\t{dataset}\tPrecision\t{precision}\t{split}\n')
        result_file.write(f'{algorithm}\t{dataset}\tSensitivity\t{sensitivity}\t{split}\n')
        result_file.write(f'{algorithm}\t{dataset}\tRecall\t{recall}\t{split}\n')
        result_file.write(f'{algorithm}\t{dataset}\tSpecificity\t{specificity}\t{split}\n')
        result_file.write(f'{algorithm}\t{dataset}\tMCC\t{MCC}\t{split}\n')
        result_file.write(f'{algorithm}\t{dataset}\tF1\t{f1_score}\t{split}\n')
    result_file.close()