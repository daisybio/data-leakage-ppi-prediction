from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from itertools import product

def calculate_auc_aupr(y_pred, y_true):
    y_pred = np.array(y_pred, dtype=float)
    y_true = np.array(y_true, dtype=int)
    auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    aupr = average_precision_score(y_true=y_true, y_score=y_pred)
    return auc, aupr


partition = False
rewired = False
multiple_runs = True
if partition:
    result_file = open("partitions/all_results.tsv", "w")
    result_file.write("Dataset\tTrain\tTest\tAUC\tAUPR\n")
    for dataset in ["du","guo","huang", "richoux", "pan", "dscript"]:
        print(f"########## Dataset: {dataset} ##########")
        for train in ["both", "0"]:
            for test in ["0", "1"]:
                if train == "0" and test == "0":
                    continue
                y_pred = []
                y_true = []
                with open(f"partitions/{dataset}_train_{train}_test_{test}.txt", "r") as f:
                    for line in f:
                        y_pred.append(line.strip().split(" ")[0])
                        y_true.append(line.strip().split(" ")[1])
                print(f"******* Train: {train}, test: {test} *******")
                auc, aupr = calculate_auc_aupr(y_pred, y_true)
                print(f"AUC: {round(auc, 4)}, AUPR: {round(aupr, 4)}")
                result_file.write(f'{dataset}\t{train}\t{test}\t{round(auc, 4)}\t{round(aupr, 4)}\n')
    result_file.close()
else:
    if rewired:
        folder = 'rewired'
        datasets = ["du", "guo", "huang", "pan", "richoux_regular", "richoux_strict", "dscript"]
    elif multiple_runs:
        folder = 'multiple_runs'
        datasets = ["guo", "huang"]
        settings = ["original", "rewired"]
        seeds = ["17612", "29715", "30940", "31191", "42446", "50495", "60688", "7413", "75212", "81645"]
        datasets = [f"{setting}_{dataset}_{seed}" for dataset, seed, setting in product(datasets, seeds, settings)]
    else:
        folder = 'original'
        datasets = ["du", "guo", "huang", "pan", "richoux_regular", "richoux_strict", "gold_standard", "dscript"]
    result_file = open(f"{folder}/all_results.tsv", "w")
    result_file.write("Dataset\tAUC\tAUPR\n")
    for dataset in datasets:
        print(f"########## Dataset: {dataset} ##########")
        y_pred = []
        y_true = []
        with open(f"{folder}/{dataset}_results.txt", "r") as f:
            for line in f:
                y_pred.append(line.strip().split(" ")[0])
                y_true.append(line.strip().split(" ")[1])
        auc, aupr = calculate_auc_aupr(y_pred, y_true)
        print(f"AUC: {round(auc, 4)}, AUPR: {round(aupr, 4)}")
        result_file.write(f'{dataset}\t{round(auc, 4)}\t{round(aupr, 4)}\n')
    result_file.close()

