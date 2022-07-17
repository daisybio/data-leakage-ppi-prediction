from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

result_file = open("all_results.tsv", "w")
result_file.write("Dataset\tTrain\tTest\tAUC\tAUPR\n")
for dataset in ["huang", "richoux", "pan"]:
    print(f"########## Dataset: {dataset} ##########")
    for train in ["both", "0"]:
        for test in ["0", "1"]:
            if train == "0" and test == "0":
                continue
            y_pred = []
            y_true = []
            with open(f"{dataset}_train_{train}_test_{test}.txt", "r") as f:
                for line in f:
                    y_pred.append(line.strip().split(" ")[0])
                    y_true.append(line.strip().split(" ")[1])
            print(f"******* Train: {train}, test: {test} *******")
            y_pred = np.array(y_pred, dtype=float)
            y_true = np.array(y_true, dtype=int)
            auc = roc_auc_score(y_true=y_true, y_score=y_pred)
            aupr = average_precision_score(y_true=y_true, y_score=y_pred)
            print(f"AUC: {round(auc, 4)}, AUPR: {round(aupr, 4)}")
            result_file.write(f'{dataset}\t{train}\t{test}\t{round(auc, 4)}\t{round(aupr, 4)}\n')

result_file.close()

