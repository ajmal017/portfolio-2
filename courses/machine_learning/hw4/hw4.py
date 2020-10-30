import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm
from sklearn.model_selection import KFold
import pandas as pd


"""
    Connor Malley
    CAP5610 - HW4
    10/06/2020
"""

def tp_tn_fp_fn(predictions, labels, positive_val=1, negative_val=0):
    # Get TP, FP, TN, and FN rates
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(predictions)):
        # True positive
        if predictions[i] == positive_val and labels[i] == positive_val:
            tp += 1
        # False positive
        elif predictions[i] == positive_val and labels[i] == negative_val:
            fp += 1
        #False negative
        elif predictions[i] == negative_val  and labels[i] == positive_val:
            fn += 1
        # True negative
        else:
            tn += 1

    return tp, tn, fp, fn


if __name__ == "__main__":
    # Task 2
    positives = np.array([[-1, -1], [1, -1]])
    negatives = np.array([[-1, 1], [1, 1]])

    plt.scatter(positives[:,0], positives[:,1], color="blue", label="Positive")
    plt.scatter(negatives[:,0], negatives[:,1], color="red", label="Negative")
    plt.xlabel("X1")
    plt.title("X1 vs. X1*X2")
    plt.ylabel("X1 * X2")
    plt.hlines(0, xmin=-1, xmax=1, linestyles="dashed", color="black")
    plt.legend()
    plt.show()

    # Task 5
    positives = np.array([[1, 1], [2, 2], [2, 0]])
    negatives = np.array([[0, 0], [1, 0], [0, 1]])
    plt.scatter(positives[:,0], positives[:,1], color="blue", label="Positive")
    plt.scatter(negatives[:,0], negatives[:,1], color="red", label="Negative")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("X1 vs. X2")
    plt.legend()
    plt.show()


    # Task 7

    # Using preprocessed titanic data from hw3
    titanic_train = pd.read_csv("titanic_preprocessed.csv")
    labels = pd.read_csv("titanic_labels.csv")

    # 5-fold cross validation
    k = 5
    kf = KFold(k, shuffle=True)
    kf.get_n_splits(titanic_train)
    tp, tn, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

    # Testing linear, quadratic, and rbf kernels
    svms = [svm.SVC(kernel='linear'), svm.SVC(kernel='poly', degree=2),
    svm.SVC(kernel='rbf')]

    # Train test loop
    for train_ind, test_ind in kf.split(titanic_train):
        xtrain, xtest = titanic_train.to_numpy()[:][train_ind], titanic_train.to_numpy()[:][test_ind]
        ytrain, ytest = labels.to_numpy()[train_ind], labels.to_numpy()[test_ind]

        for i in range(len(svms)):
            svms[i].fit(xtrain, ytrain.ravel())

            this_tp, this_tn, this_fp, this_fn = tp_tn_fp_fn(svms[i].predict(xtest), ytest.ravel())
            tp[i] += this_tp
            tn[i] += this_tn
            fp[i] += this_fp
            fn[i] += this_fn

    # Getting accuracies of different kernels
    accuracy = np.zeros(3)
    for i in range(len(svms)):
        accuracy[i] = (tp[i] + tn[i]) / (tp[i] + fp[i] + fn[i] + tn[i])


    print("Linear Kernel Accuracy (5-fold): %f" % accuracy[0]) 
    print("Quadratic Kernel Accuracy (5-fold): %f" % accuracy[1]) 
    print("RBF Kernel Accuracy (5-fold): %f" % accuracy[2]) 