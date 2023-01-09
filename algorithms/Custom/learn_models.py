import numpy as np

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


