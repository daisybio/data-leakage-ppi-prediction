from load_datasets import load_partition_datasets, load_from_SPRINT
from learn_models import learn_rf, learn_SVM
from time import time


def export_scores(scores, path):
    with open(path, 'w') as f:
        for key, value in scores.items():
            f.write(f'{key},{value}\n')


def load_data(name, encoding, partition=False, partition_train='0', partition_test='1', rewire=False):
    if partition:
        print(f"Loading partition datasets for {name}, train on {partition_train}, test on {partition_test}")
        X_train, y_train, X_test, y_test = load_partition_datasets(encoding=encoding, dataset=name,
                                                                   partition_train=partition_train,
                                                                   partiton_test=partition_test)
    else:
        X_train, y_train, X_test, y_test = load_from_SPRINT(encoding=encoding, dataset=name, rewire=rewire)
    print(f'Train: {sum(y_train)} positive PPIs, {len(y_train) - sum(y_train)} negative PPIs, all: {len(y_train)}')
    print(f'Test: {sum(y_test)} positive PPIs, {len(y_test) - sum(y_test)} negative PPIs, all: {len(y_test)}')
    return X_train, y_train, X_test, y_test


def run_partitioning_tests():
    for name in ['guo', 'huang', 'du', 'pan', 'richoux']:
        for encoding in ['PCA', 'MDS', 'node2vec']:
            for partition_train in ['both', '0']:
                for partition_test in ['0', '1']:
                    if partition_train == '0' and partition_test == '0':
                        continue
                    else:
                        t_start = time()
                        print(
                            f'##### {name} dataset, {encoding} encoding, train: {partition_train}, test: {partition_test}')
                        X_train, y_train, X_test, y_test = load_data(name=name, encoding=encoding,
                                                                     partition=True, partition_train=partition_train,
                                                                     partition_test=partition_test, rewire=False)
                        time_preprocess = time() - t_start
                        scores = learn_rf(X_train, y_train, X_test, y_test)
                        export_scores(scores,
                                      f'results/partition_tests/{name}_{encoding}_RF_{partition_train}_{partition_test}.csv')
                        time_elapsed_rf = time() - t_start
                        print(f'time elapsed: {time_elapsed_rf}')
                        with open(f'results/time_partition_{name}_{encoding}.txt', 'a+') as f:
                            f.write(f'{partition_train}\t{partition_test}\tRF\t{time_elapsed_rf}\n')

                        scores = learn_SVM(X_train, y_train, X_test, y_test)
                        export_scores(scores,
                                      f'results/partition_tests/{name}_{encoding}_SVM_{partition_train}_{partition_test}.csv')
                        time_elapsed_svm = time() - t_start - time_elapsed_rf + time_preprocess
                        print(f'time elapsed: {time_elapsed_svm}')
                        with open(f'results/time_partition_{name}_{encoding}.txt', 'a') as f:
                            f.write(f'{partition_train}\t{partition_test}\tSVM\t{time_elapsed_svm}\n')


def run_simpler_algorithms(rewire=False):
    if rewire:
        prefix = 'rewired_'
    else:
        prefix = 'original_'
    dataset_list = ['guo', 'huang', 'du', 'pan', 'richoux_regular', 'richoux_strict']
    for name in dataset_list:
        for encoding in ['PCA', 'MDS', 'node2vec']:
            t_start = time()
            print(
                f'##### {name} dataset, {encoding} encoding')
            X_train, y_train, X_test, y_test = load_data(name=name, encoding=encoding,
                                                         partition=False, rewire=rewire)
            time_preprocess = time() - t_start
            scores = learn_rf(X_train, y_train, X_test, y_test)
            export_scores(scores,
                          f'results/{prefix}{name}_{encoding}_RF.csv')
            time_elapsed_rf = time() - t_start
            print(f'time elapsed: {time_elapsed_rf}')
            with open(f'results/time_{prefix}{name}_{encoding}.txt', 'w') as f:
                f.write(f'RF\t{time_elapsed_rf}\n')
            scores = learn_SVM(X_train, y_train, X_test, y_test)
            export_scores(scores,
                          f'results/{prefix}{name}_{encoding}_SVM.csv')
            time_elapsed_svm = time() - t_start - time_elapsed_rf + time_preprocess
            print(f'time elapsed: {time_elapsed_svm}')
            with open(f'results/time_{prefix}{name}_{encoding}.txt', 'a') as f:
                f.write(f'SVM\t{time_elapsed_svm}')


if __name__ == "__main__":
    print('########################### ORIGINAL ###########################')
    run_simpler_algorithms(rewire=False)
    print('########################### REWIRED ###########################')
    run_simpler_algorithms(rewire=True)
    print('########################### PARTITION ###########################')
    run_partitioning_tests()
