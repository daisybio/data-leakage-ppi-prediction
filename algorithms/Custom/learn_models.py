import numpy as np
from networkx.algorithms import node_classification


def calculate_scores(y_true, y_pred):
    from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
    # Print out the scores
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp + tn) / (tp + fp + tn + fn)
    prec = tp / (tp + fp)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    print('Accuracy:', round(acc, 4))
    print('Precision:', round(prec, 4))
    print('Sensitivity/Recall:', round(sens, 4))
    print('Specificity:', round(spec, 4))
    print('F1:', round(f1, 4))
    print('MCC:', round(mcc, 4))
    return {'Accuracy': acc,
            'Precision': prec,
            'Sensitivity': sens,
            'Specificity': spec,
            'F1': f1,
            'MCC': mcc}


def learn_rf(train_features, train_labels, test_features, test_labels):
    from sklearn.ensemble import RandomForestClassifier
    # Instantiate model with 100 decision trees
    rf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1, n_jobs=6)
    # Train the model on training data
    print("Fitting RF ...")
    rf.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    print("Predicting ...")
    y_pred = rf.predict(test_features)
    #y_pred = np.array(np.where(y_pred > 0.5, 1, 0), dtype=int)
    return calculate_scores(y_true=test_labels, y_pred=y_pred)


def learn_SVM(train_features, train_labels, test_features, test_labels):
    import numpy as np
    from sklearn.svm import SVC
    clf = SVC(random_state=42)
    print("Fitting SVM ...")
    clf.fit(train_features, train_labels)
    print("Predicting ...")
    y_pred = clf.predict(test_features)
    y_pred = np.array(np.where(y_pred > 0.5, 1, 0), dtype=int)
    return calculate_scores(y_true=test_labels, y_pred=y_pred)


def semi_supervised_analysis(line_graph, shuffle_labels, rewired, method_name):
    import networkx as nx
    # get node_ids dict (uniprotid1, uniprotid2): enumerated node_id and node_labels dict node_id: interaction label
    node_ids, node_labels = _get_ids_and_labels_of_labeled_nodes(line_graph, 'interaction')
    training_nodes = [node_ids[node] for node, data in line_graph.nodes(data=True) if data['split'] == 'training']
    test_nodes = [node_ids[node] for node, data in line_graph.nodes(data=True) if data['split'] == 'test']
    if shuffle_labels:
        node_labels = _shuffle_labels(node_labels)
    y_true = np.array([node_labels[node_id] for node_id in node_ids.values() if node_id in test_nodes], dtype=int)
    tmp_lg = nx.Graph(line_graph)
    if rewired:
        degree_sequence = [d for _, d in tmp_lg.degree()]
        tmp_lg = nx.expected_degree_graph(degree_sequence, selfloops=False)
    node_list = list(tmp_lg.nodes)
    for node_id in training_nodes:
        node = node_list[node_id]
        tmp_lg.nodes[node]['label'] = node_labels[node_id]
    predicted_labels = _get_method(method_name)(tmp_lg)
    y_pred = np.array(predicted_labels, dtype=int)[test_nodes]
    scores = calculate_scores(y_true, y_pred)
    return scores


def _get_ids_and_labels_of_labeled_nodes(line_graph, predict_attribute):
    node_ids = dict()
    node_labels = dict()
    for node_id, node in enumerate(line_graph.nodes(data=True)):
        label = node[1][predict_attribute]
        node_ids[node[0]] = node_id
        node_labels[node_id] = label
    return node_ids, node_labels


def _get_method(method_name):
    method_dict = {
        'Harmonic function': node_classification.harmonic_function,
        'Local and global consistency': node_classification.local_and_global_consistency
    }
    return method_dict[method_name]


def _shuffle_labels(node_labels):
    import numpy as np
    node_ids = list(node_labels.keys())
    labels = list(node_labels.values())
    np.random.shuffle(labels)
    for i in range(len(node_labels)):
        node_labels[node_ids[i]] = labels[i]
    return node_labels