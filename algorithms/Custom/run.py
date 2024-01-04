import sys
from load_datasets import load_partition_datasets, load_from_SPRINT, load_gold_standard, construct_line_graph
from learn_models import learn_rf, learn_SVM, semi_supervised_analysis
from time import time


def export_scores(scores, path):
    with open(path, 'w') as f:
        for key, value in scores.items():
            f.write(f'{key},{value}\n')


def load_data(name, encoding, partition=False, partition_train='0', partition_test='1', rewire=False, seed=None):
    if name == 'gold_standard_':
        print('loading gold standard ...')
        X_train, y_train, X_test, y_test = load_gold_standard(encoding=encoding)
    elif name == 'gold_standard_unbalanced_':
        print('loading gold standard unbalanced ...')
        X_train, y_train, X_test, y_test = load_gold_standard(encoding=encoding, unbalanced=True)
    elif partition:
        print(f"Loading partition datasets for {name}, train on {partition_train}, test on {partition_test}")
        X_train, y_train, X_test, y_test = load_partition_datasets(encoding=encoding, dataset=name,
                                                                   partition_train=partition_train,
                                                                   partiton_test=partition_test)
    else:
        X_train, y_train, X_test, y_test = load_from_SPRINT(encoding=encoding, dataset=name, rewire=rewire, seed=seed)
    print(f'Train: {sum(y_train)} positive PPIs, {len(y_train) - sum(y_train)} negative PPIs, all: {len(y_train)}')
    print(f'Test: {sum(y_test)} positive PPIs, {len(y_test) - sum(y_test)} negative PPIs, all: {len(y_test)}')
    return X_train, y_train, X_test, y_test


def run_partitioning_tests(dataset_list=None):
    if dataset_list is None:
        dataset_list = ['guo', 'huang', 'du', 'pan', 'richoux', 'dscript']
    for name in dataset_list:
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


def run_simpler_algorithms(rewire=False, dataset_list=None, seed=None):
    if dataset_list is None:
        dataset_list = ['guo', 'huang', 'du', 'pan', 'richoux_regular', 'richoux_strict', 'dscript']
    if rewire:
        prefix = 'rewired_'
    else:
        prefix = 'original_'
    for name in dataset_list:
        for encoding in ['PCA', 'MDS', 'node2vec']:
            t_start = time()
            print(
                f'##### {name} dataset, {encoding} encoding')
            X_train, y_train, X_test, y_test = load_data(name=name, encoding=encoding,
                                                         partition=False, rewire=rewire,
                                                         seed=seed)
            time_preprocess = time() - t_start
            scores = learn_rf(X_train, y_train, X_test, y_test)
            if seed is not None:
                export_scores(scores, f'results/multiple_runs/{prefix}{name}_{encoding}_RF_{seed}.csv')
            else:
                export_scores(scores, f'results/{prefix}{name}_{encoding}_RF.csv')
            time_elapsed_rf = time() - t_start
            print(f'time elapsed: {time_elapsed_rf}')
            if seed is not None:
                with open(f'results/multiple_runs/time_{prefix}{name}_{encoding}_{seed}.txt', 'a+') as f:
                    f.write(f'RF\t{time_elapsed_rf}\n')
            else:
                with open(f'results/time_{prefix}{name}_{encoding}.txt', 'a+') as f:
                    f.write(f'RF\t{time_elapsed_rf}\n')
            scores = learn_SVM(X_train, y_train, X_test, y_test)
            if seed is not None:
                export_scores(scores,
                              f'results/multiple_runs/{prefix}{name}_{encoding}_SVM_{seed}.csv')
            else:
                export_scores(scores,
                              f'results/{prefix}{name}_{encoding}_SVM.csv')
            time_elapsed_svm = time() - t_start - time_elapsed_rf + time_preprocess
            print(f'time elapsed: {time_elapsed_svm}')
            if seed is not None:
                with open(f'results/multiple_runs/time_{prefix}{name}_{encoding}_{seed}.txt', 'a') as f:
                    f.write(f'SVM\t{time_elapsed_svm}\n')
            else:
                with open(f'results/time_{prefix}{name}_{encoding}.txt', 'a') as f:
                    f.write(f'SVM\t{time_elapsed_svm}')


def run_degree_algorithm(rewire=False, partition=False, gold=False, unbalanced=False, dataset_list=None, seed=None):
    if rewire:
        prefix = 'rewired_'
        if dataset_list is None:
            dataset_list = ['guo', 'huang', 'du', 'pan', 'richoux_regular', 'richoux_strict', 'dscript']
    elif partition:
        prefix = 'partition_'
        if dataset_list is None:
            dataset_list = ['guo_both_0','guo_both_1','guo_0_1',
                    'huang_both_0', 'huang_both_1', 'huang_0_1',
                    'du_both_0', 'du_both_1', 'du_0_1',
                    'pan_both_0', 'pan_both_1', 'pan_0_1',
                    'richoux_both_0', 'richoux_both_1', 'richoux_0_1',
                    'dscript_both_0', 'dscript_both_1', 'dscript_0_1']
        else:
            # append "_both_0", "_both_1", "_0_1" to all dataset_list names
            dataset_list = [f'{name}_both_0' for name in dataset_list] + [f'{name}_both_1' for name in dataset_list] + [f'{name}_0_1' for name in dataset_list]
    elif gold:
        prefix = 'gold_standard_'
        dataset_list = ['gold_standard']
    elif unbalanced:
        prefix = 'gold_standard_unbalanced_'
        dataset_list = ['gold_standard_unbalanced']
    else:
        prefix = 'original_'
        if dataset_list is None:
            dataset_list = ['guo', 'huang', 'du', 'pan', 'richoux_regular', 'richoux_strict', 'dscript']
    for name in dataset_list:
        t_start = time()
        print(f'##### degree algorithm: {name} dataset #####')
        lg = construct_line_graph(dataset=name, prefix=prefix, seed=seed)
        print('Constructed line graph!')
        time_preprocess = time() - t_start
        #try:
        #    scores_hf = semi_supervised_analysis(lg, shuffle_labels=False, rewired=False, method_name='Harmonic function')
        #except:
        #    print('Harmonic function failed!')
        #    scores_hf = {}
        #if seed is not None:
        #    export_scores(scores_hf,
        #                  f'results/multiple_runs/{prefix}{name}_hf_{seed}.csv')
        #else:
        #    export_scores(scores_hf,
        #                  f'results/{prefix}{name}_hf.csv')
        time_elapsed_hf = time() - t_start
        print(f'time elapsed: {time_elapsed_hf}')
        scores_cons = semi_supervised_analysis(lg, shuffle_labels=False, rewired=False, method_name='Local and global consistency')
        if seed is not None:
            export_scores(scores_cons,
                          f'results/multiple_runs/{prefix}{name}_cons_{seed}.csv')
        else:
            export_scores(scores_cons,
                      f'results/{prefix}{name}_cons.csv')
        time_elapsed_cons = time() - t_start - time_elapsed_hf + time_preprocess
        print(f'time elapsed: {time_elapsed_cons}')
        if seed is not None:
            with open(f'results/multiple_runs/time_{prefix}deg_{seed}.txt', 'a+') as f:
                f.write(f'{name}\tHarmonic Function\t{time_elapsed_hf}\n')
                f.write(f'{name}\tLocal and Global Consistency\t{time_elapsed_cons}\n')
        else:
            with open(f'results/time_{prefix}deg.txt', 'a+') as f:
                f.write(f'{name}\tHarmonic Function\t{time_elapsed_hf}\n')
                f.write(f'{name}\tLocal and Global Consistency\t{time_elapsed_cons}\n')


def run_gold_standard(unbalanced=False):
    if unbalanced:
        prefix = 'gold_standard_unbalanced_'
    else:
        prefix = 'gold_standard_'
    for encoding in ['PCA', 'MDS', 'node2vec']:
        t_start = time()
        print(f'##### {encoding} encoding')
        X_train, y_train, X_test, y_test = load_data(name=prefix, encoding=encoding,
                                                     partition=False, rewire=False)
        time_preprocess = time() - t_start
        scores = learn_rf(X_train, y_train, X_test, y_test)
        export_scores(scores,
                      f'results/{prefix}{encoding}_RF.csv')
        time_elapsed_rf = time() - t_start
        print(f'time elapsed: {time_elapsed_rf}')
        with open(f'results/time_{prefix}{encoding}.txt', 'a+') as f:
            f.write(f'RF\t{time_elapsed_rf}\n')
        scores = learn_SVM(X_train, y_train, X_test, y_test)
        export_scores(scores,
                      f'results/{prefix}{encoding}_SVM.csv')
        time_elapsed_svm = time() - t_start - time_elapsed_rf + time_preprocess
        print(f'time elapsed: {time_elapsed_svm}')
        with open(f'results/time_{prefix}{encoding}.txt', 'a') as f:
            f.write(f'SVM\t{time_elapsed_svm}')


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 1:
        dataset_list = [arg for arg in args[1].split(',')]
        print(f'Using dataset list {dataset_list}')
        if len(args) > 2:
            seed = int(args[2])
            print(f'Using seed {seed}')
        else:
            seed = None
    else:
        seed = None
        dataset_list = None
    if args[0] == 'original':
        print('########################### ORIGINAL ###########################')
        run_simpler_algorithms(rewire=False, dataset_list=dataset_list, seed=seed)
        run_degree_algorithm(rewire=False, partition=False, dataset_list=dataset_list, seed=seed)
    elif args[0] == 'rewired':
        print('########################### REWIRED ###########################')
        run_simpler_algorithms(rewire=True, dataset_list=dataset_list, seed=seed)
        run_degree_algorithm(rewire=True, partition=False, dataset_list=dataset_list, seed=seed)
    elif args[0] == 'partition':
        print('########################### PARTITION ###########################')
        #run_partitioning_tests(dataset_list=dataset_list)
        run_degree_algorithm(rewire=False, partition=True)
    elif args[0] == 'gold_standard':
        print('########################### GOLD STANDARD ###########################')
        #run_gold_standard()
        run_degree_algorithm(rewire=False, partition=False, gold=True)
    else:
        print('########################### GOLD STANDARD UNBALANCED ###########################')
        run_gold_standard(unbalanced=True)
        run_degree_algorithm(rewire=False, partition=False, unbalanced=True)
