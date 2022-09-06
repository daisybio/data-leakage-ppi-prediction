from load_datasets import load_partition_datasets, load_from_SPRINT
from learn_models import learn_rf, learn_SVM
from time import time


def export_scores(scores, path):
    with open(path, 'w') as f:
        for key, value in scores.items():
            f.write(f'{key},{value}\n')


def load_data(name, encoding, dataset='regular', partition=False, partition_train='0', partition_test='1', rewire=False):
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


def run_partitioning_tests(rewire=False):
    for name in ['du', 'guo', 'huang', 'pan', 'richoux']:
        for encoding in ['PCA', 'MDS', 'node2vec']:
            for partition_train in ['both', '0']:
                for partition_test in ['0', '1']:
                    if partition_train == '0' and partition_test == '0':
                        continue
                    else:
                        print(
                            f'##### {name} dataset, {encoding} encoding, train: {partition_train}, test: {partition_test}')
                        X_train, y_train, X_test, y_test = load_data(name=name, encoding=encoding,
                                                                     partition=True, partition_train=partition_train,
                                                                     partition_test=partition_test, rewire=rewire)
                        scores = learn_rf(X_train, y_train, X_test, y_test)
                        export_scores(scores,
                                      f'results/partition_tests/{name}_{encoding}_RF_tr{partition_train}_test_{partition_test}.csv')
                        scores = learn_SVM(X_train, y_train, X_test, y_test)
                        export_scores(scores,
                                      f'results/partition_tests/{name}_{encoding}_SVM_tr{partition_train}_test_{partition_test}.csv')


def run_simpler_algorithms(rewire=False):
    if rewire:
        prefix = 'rewired_'
    else:
        prefix = ''
    dataset_list = ['huang', 'guo', 'du', 'pan', 'richoux_regular', 'richoux_strict']
    for name in dataset_list:
        for encoding in ['node2vec']:
            if name == 'richoux':
                for dataset in ['regular', 'strict']:
                    #t_start = time()
                    print(
                        f'##### {name} dataset: {dataset}, {encoding} encoding')
                    X_train, y_train, X_test, y_test = load_data(name=name, encoding=encoding,
                                                                 partition=False, dataset=dataset, rewire=rewire)
                    #time_preprocess = time() - t_start
                    scores = learn_rf(X_train, y_train, X_test, y_test)
                    #export_scores(scores,
                    #              f'results/{prefix}{name}_{encoding}_{dataset}_RF.csv')
                    #time_elapsed_rf = time() - t_start
                    print(f'time elapsed: {time_elapsed_rf}')
                    #with open( f'results/time_{prefix}{name}_{encoding}_{dataset}.txt', 'w') as f:
                    #    f.write(f'RF\t{time_elapsed_rf}\n')
                    #scores = learn_SVM(X_train, y_train, X_test, y_test)
                    #export_scores(scores,
                    #              f'results/{prefix}{name}_{encoding}_{dataset}_SVM.csv')
                    #time_elapsed_svm = time() - t_start - time_elapsed_rf + time_preprocess
                    #print(f'time elapsed: {time_elapsed_svm}')
                    #with open(f'results/time_{prefix}{name}_{encoding}_{dataset}.txt', 'a') as f:
                    #    f.write(f'SVM\t{time_elapsed_svm}')

            else:
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
    run_simpler_algorithms(rewire=True)
    #run_partitioning_tests()
