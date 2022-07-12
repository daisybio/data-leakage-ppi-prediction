from load_datasets import load_du, load_guo, load_pan, load_richoux, load_partition_datasets
from learn_models import learn_rf, learn_SVM


def export_scores(scores, path):
    with open(path, 'w') as f:
        for key, value in scores.items():
            f.write(f'{key},{value}\n')


def load_data(name, encoding, dataset='regular', partition=False, partition_train='0', partition_test='1'):
    if partition:
        print(f"Loading partition datasets for {name}, train on {partition_train}, test on {partition_test}")
        X_train, y_train, X_test, y_test = load_partition_datasets(encoding=encoding, dataset=name,
                                                                   partition_train=partition_train,
                                                                   partiton_test=partition_test)
    elif name == 'du':
        print("Loading Du et al. yeast dataset ...")
        X_train, y_train, X_test, y_test = load_du(encoding=encoding)
    elif name == 'guo':
        print("Loading Guo et al. yeast dataset ...")
        X_train, y_train, X_test, y_test = load_guo(encoding=encoding)
    elif name == 'pan':
        print('Loading Pan et al. human dataset ...')
        X_train, y_train, X_test, y_test = load_pan(encoding=encoding)
    elif name == 'richoux':
        print('Loading Richoux et al. human dataset ...')
        X_train, y_train, X_test, y_test = load_richoux(encoding=encoding, dataset=dataset)
    print(f'Train: {sum(y_train)} positive PPIs, {len(y_train) - sum(y_train)} negative PPIs, all: {len(y_train)}')
    print(f'Test: {sum(y_test)} positive PPIs, {len(y_test) - sum(y_test)} negative PPIs, all: {len(y_test)}')
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
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
                                                                     partition_test=partition_test)
                        scores = learn_rf(X_train, y_train, X_test, y_test)
                        export_scores(scores,
                                      f'results/partition_tests/{name}_{encoding}_RF_tr{partition_train}_test_{partition_test}.csv')
                        scores = learn_SVM(X_train, y_train, X_test, y_test)
                        export_scores(scores,
                                      f'results/partition_tests/{name}_{encoding}_SVM_tr{partition_train}_test_{partition_test}.csv')
