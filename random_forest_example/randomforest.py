import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import roc_curve, auc

def get_dataset_from_csv(filename):
    """Retrieves data from csv file and stores it in numpy arrays
    Takes filename
    Returns x (raw data), y (indexed labels)
    """
    x = None
    y = None
    # Open data file
    with open(filename, newline='') as stream:
        # On successfully opening a stream, count number of lines
        num_lines = sum(1 for line in stream)
        # Reset filestream to beginning of file
        stream.seek(0)
        # Instantiate csv reader for file
        reader = csv.reader(stream)

        # Read first line of file to get shape and instantiate arrays
        row = next(reader)
        x = np.zeros((num_lines, len(row)-1), dtype=np.float32)
        y = np.zeros(num_lines, dtype=np.int32)
        # Add first line to dataset
        i = 0
        x[i,:] = np.array(row[:-1]).astype(np.float32)
        y[i] = int(row[-1].startswith('m'))
        i = i + 1
        # Add rest of lines to dataset
        for row in reader:
            x[i,:] = np.array(row[:-1]).astype(np.float32)
            y[i] = int(row[-1].startswith('m'))
            i = i + 1
    return x, y

def normalize_columns(data):
    """Normalize the features to have mean 0 and standard deviation 1
    This is done for numerical stability
    """
    stds = np.std(data, axis=0)
    mus = np.mean(data, axis=0)
    return np.divide(np.subtract(data, mus),stds)


# Execute this if this file is run as __main__
if __name__ == '__main__':
    # First command line argument is filename
    data, labels = get_dataset_from_csv("data.csv")
    if data is None:
        print("Error reading file %s" % sys.argv[1])
        sys.exit(1)

    # Normalize data
    data = normalize_columns(data)

    n_trials = 10
    # Train n_trials random forests and report mean and stdev for accuracy
    accuracies = []
    aucs = []
    importances = np.zeros((n_trials,data.shape[1]))

    # Compute ROC curve and ROC area for each class
    roc_auc = dict()

    for i in range(0, n_trials):
        (
            X_train, X_test, y_train, y_test
        ) = train_test_split(data, labels, test_size=0.1)

        # Train random forest
        classifier = RandomForestClassifier(n_estimators=20)
        classifier.fit(X_train, y_train)
        importances[i,:] = classifier.feature_importances_
        accuracies = accuracies + [accuracy(y_test, classifier.predict(X_test))]

        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_score[:,1], pos_label=1)
        aucs  = aucs + [auc(fpr, tpr)]


    print("Mean Accuracy: %0.3f\nStandard Deviation: %0.3f" % (np.mean(accuracies), np.std(accuracies)))
    print("Importances: ", np.mean(importances, axis=0))
    print("Mean AUC: %f" % np.mean(aucs))


    fpr, tpr, _ = roc_curve(y_test, y_score[:,1], pos_label=1)
    aucs  = aucs + [auc(fpr, tpr)]


    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % aucs[-1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC For Malignant Class')
    plt.legend(loc="lower right")
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_score[:,0], pos_label=0)
    aucs  = aucs + [auc(fpr, tpr)]


    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % aucs[-1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC For Benign Class')
    plt.legend(loc="lower right")
    plt.show()
