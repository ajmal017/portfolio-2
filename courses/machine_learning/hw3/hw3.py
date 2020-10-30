import sklearn
from sklearn.naive_bayes import CategoricalNB, MultinomialNB, GaussianNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import pandas as pd
import datetime
import numpy as np
import sys
import math
import operator
import matplotlib.pyplot as plt


class CategoricalEncoder(object):
    def __init__(self, fill_val=0, start=1):
        self.fill_val = fill_val
        self.start = start

    def fit(self, X):
        self.classes = {}
        class_counter = self.start
        for key in X:
            if key not in self.classes:
                self.classes[key] = class_counter
                class_counter += 1

    def transform(self, X):
        arr = np.zeros(len(X))
        for i in range(len(X)):
            if (X[i] in self.classes):
                arr[i] = self.classes[X[i]]
            else:
                arr[i] = self.fill_val

        return arr

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

# Euclidian distance of 2 array-like objects
def euclidean_distance(x1, x2, weights=None):
    dist = 0
    for i in range(len(x1)):
        if not weights:
            dist += pow(x1[i]-x2[i], 2)
        else:
            dist += weights[i]*pow(x1[i]-x2[i], 2)

    return math.sqrt(dist)

def getKNearestNeighbors(dataset, reference, k):
    distances = []
    for i in range(len(dataset[:,0])):
        distances.append((dataset[i][:], i, euclidean_distance(dataset[i][:], reference)))


    distances.sort(key=operator.itemgetter(2))

    k_neighbors = []
    for i in range(0,k):
        k_neighbors.append(distances[i][1])

    return k_neighbors


def get_vote(indices, labels, weighted=False):
    classes = {}
    for ind in indices:
        if labels[ind] in classes:
            classes[labels[ind]] += 1
        else:
            classes[labels[ind]] = 1

    max_val   = 0 
    max_label = 0
    for label in classes.keys():
        if classes[label] >= max_val:
            max_val = classes[label]
            max_label = label

    return max_label

def knn_predict(training_features, training_labels, testing_set, k=5):
    predictions = []
    for i in range(len(testing_set)):
        k_neighbors = getKNearestNeighbors(training_features, testing_set[i], k=k)
        predictions.append(get_vote(k_neighbors, training_labels))
    
    return predictions




if __name__ == "__main__":
    train_data = pd.read_csv("./train_games.csv", delimiter="\t")
    test_data  = pd.read_csv("./test_games.csv", delimiter="\t")

    train_data = train_data.drop("ID", axis=1)
    train_data = train_data.drop("Date", axis=1)
    test_data = test_data.drop("ID", axis=1)
    test_data = test_data.drop("Date", axis=1)

    train_encoded = pd.DataFrame()
    test_encoded = pd.DataFrame()

    # Encode the categorical data
    ce = CategoricalEncoder()
    le = LabelEncoder()
    columns = list(train_data.to_dict().keys())
    for column in columns:
        if column != "ID" and column != "Date" and column != "Label":
            ce.fit(train_data[column])
            # Using pd dummies in case category in test set isnt present in train
            train_data[column] = ce.transform(train_data[column])
            test_data[column] = ce.transform(test_data[column])


    # Saves off encoded labels separately
    train_labels = train_data['Label']
    test_labels  = test_data['Label']

    # Dropping unneccessary features (and labels)
    train_data = train_data.drop("Label", axis=1)
    test_data = test_data.drop("Label", axis=1)

    columns = list(train_data.to_dict().keys())

    # Create NB model
    clf = CategoricalNB()
    clf.fit(train_data[columns], train_labels)

    # Get TP, FP, TN, and FN rates
    nb_predictions = clf.predict(test_data[columns])
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(nb_predictions)):
        # True positive
        if nb_predictions[i] == 'Win' and test_labels[i] == 'Win':
            tp += 1
        # False positive
        elif nb_predictions[i] == 'Win' and test_labels[i] == 'Lose':
            fp += 1
        #False negative
        elif nb_predictions[i] == 'Lose'  and test_labels[i] == 'Win':
            fn += 1
        # True negative
        else:
            tn += 1


    #print(f"Naive Bayes Predictions: %s" % str(nb_predictions))
    print(list(test_labels))

    # Gets accuracy
    nb_accuracy = (tp + tn) / (tp + fp + fn + tn)
    print("Naive Bayes Accuracy: %f" % nb_accuracy)

    # Gets precision
    nb_precision = tp / (tp + fp)
    print("Naive Bayes Precision: %f" % nb_precision)

    # Gets recall
    nb_recall = tp / (tp + fn)
    print("Naive Bayes Recall: %f" % nb_recall)

    # Gets F1
    nb_f1 = 2*(nb_recall * nb_precision) / (nb_recall + nb_precision)
    print("Naive Bayes F1: %f" % nb_f1)

    # Initializing encoded data
    encoded_train_data = pd.get_dummies(train_data[train_data.keys()[0]])
    encoded_test_data  = pd.get_dummies(test_data[test_data.keys()[0]]).reindex(columns=encoded_train_data.columns, fill_value=0).to_numpy()
    encoded_train_data = encoded_train_data.to_numpy()

    # Need to one hot encode data for KNN
    for column in train_data.columns[1:]:
        train_dummies = pd.get_dummies(train_data[column])
        test_dummies  = pd.get_dummies(test_data[column]).reindex(columns=train_dummies.columns, fill_value=0).to_numpy()
        train_dummies = train_dummies.to_numpy()
        encoded_train_data = np.append(encoded_train_data, train_dummies, axis=1)
        encoded_test_data = np.append(encoded_test_data, test_dummies, axis=1)

        
    print(encoded_train_data)
    print(encoded_test_data)
    print(np.shape(encoded_train_data))

    k_list = np.arange(10)
    print(k_list)

    football_knn = KNeighborsClassifier(n_neighbors=3)
    football_knn.fit(encoded_train_data, train_labels)
    knn_predictions = football_knn.predict(encoded_test_data)
    print("KNN Predictions: %s" % str(knn_predictions))
    print(list(test_labels))

    tp, tn, fp, fn = tp_tn_fp_fn(knn_predictions, test_labels, positive_val='Win', negative_val='Lose')

    # Gets accuracy
    knn_accuracy = (tp + tn) / (tp + fp + fn + tn)
    print("KNN Accuracy: %f" % knn_accuracy)

    # Gets precision
    knn_precision = tp / (tp + fp)
    print("KNN Precision: %f" % knn_precision)

    # Gets recall
    knn_recall = tp / (tp + fn)
    print("KNN Recall: %f" % knn_recall)

    # Gets F1
    knn_f1 = 2*(knn_recall * knn_precision) / (knn_recall + knn_precision)
    print("KNN F1: %f" % knn_f1)


    # Titanic Code
    ############################################
    titanic_train = pd.read_csv("./titanic_train.csv")
    titanic_test  = pd.read_csv("./titanic_test.csv")

    labels = titanic_train['Survived']

    titanic_train = titanic_train[['Sex', 'Age', 'Embarked', 'Pclass']]

    # Replacing nan ages with mean
    nan_ages = np.isnan(titanic_train['Age'])
    mean_age = np.mean(titanic_train['Age'])
    titanic_train['Age'][nan_ages] = mean_age

    # Putting the age data into bins
    age_bins = np.linspace(0,100, 8, endpoint=True)
    titanic_train['Age'] = np.digitize(titanic_train['Age'], age_bins)

    # Data should all be categorical now, just encode age and embarked
    titanic_train['Sex']      = le.fit_transform(titanic_train['Age'])
    titanic_train['Embarked'] = le.fit_transform(titanic_train['Age'])
    print(titanic_train)

    titanic_train.to_csv("titanic_preprocessed.csv", index=False)
    labels_df = pd.DataFrame(labels)
    labels_df.to_csv("titanic_labels.csv", index=False)

    k = 5
    kf = KFold(k, shuffle=True)
    kf.get_n_splits(titanic_train)

    titanic_nb = CategoricalNB()

    tp, tn, fp, fn = 0, 0, 0, 0
    for train_ind, test_ind in kf.split(titanic_train):
        xtrain, xtest = titanic_train.to_numpy()[:][train_ind], titanic_train.to_numpy()[:][test_ind]
        ytrain, ytest = labels.to_numpy()[train_ind], labels.to_numpy()[test_ind]


        titanic_nb.fit(xtrain, ytrain)

        this_tp, this_tn, this_fp, this_fn = tp_tn_fp_fn(titanic_nb.predict(xtest), ytest)
        tp += this_tp
        tn += this_tn
        fp += this_fp
        fn += this_fn


    tp /= k
    tn /= k
    fp /= k
    fn /= k

    # Gets accuracy
    nb_accuracy = (tp + tn) / (tp + fp + fn + tn)
    print("Naive Bayes Accuracy: %f" % nb_accuracy)
    # Gets precision
    nb_precision = tp / (tp + fp)
    print("Naive Bayes Precision: %f" % nb_precision)
    # Gets recall
    nb_recall = tp / (tp + fn)
    print("Naive Bayes Recall: %f" % nb_recall)
    # Gets F1
    nb_f1 = 2*(nb_recall * nb_precision) / (nb_recall + nb_precision)
    print("Naive Bayes F1: %f" % nb_f1)

    print(titanic_train)

    k_nearest = np.arange(1, 100)
    knn_accuracy  = np.zeros(len(k_nearest))
    knn_precision = np.zeros(len(k_nearest))
    knn_recall    = np.zeros(len(k_nearest))
    knn_f1        = np.zeros(len(k_nearest))

    mm_scaler = sklearn.preprocessing.MinMaxScaler()

    # Normalizing between 0 and 1 
    train_values = titanic_train.values
    train_scaled = mm_scaler.fit_transform(train_values)
    titanic_train = pd.DataFrame(train_scaled)

    for i in range(len(k_nearest)):
        print("Trying K = %d" % (i+1))
        for train_ind, test_ind in kf.split(titanic_train):
            xtrain, xtest = titanic_train.to_numpy()[:][train_ind], titanic_train.to_numpy()[:][test_ind]
            ytrain, ytest = labels.to_numpy()[train_ind], labels.to_numpy()[test_ind]
            
            predictions = knn_predict(xtrain, ytrain, xtest, k=k_nearest[i]) 
            this_tp, this_tn, this_fp, this_fn = tp_tn_fp_fn(predictions, ytest)
            print("fp / fn ratio: %f" % (this_fp/this_fn))
            if (this_tp + this_fp) == 0:
                print("k == %d, tp = %d, fp = %d" % (k_nearest[i], tp, fp))
            this_knn_accuracy  = (this_tp + this_tn) / (this_tp + this_fp + this_fn + this_tn)
            if (this_tp + this_fp) != 0:
                this_knn_precision = this_tp / (this_tp + this_fp)
            else:
                this_knn_precision = 0
            this_knn_recall    = this_tp / (this_tp + this_fn)
            
            knn_f1[i] += 2*(this_knn_recall * this_knn_precision) / (this_knn_recall + this_knn_precision)
            knn_recall[i] += this_knn_recall
            knn_precision[i] += this_knn_precision
            knn_accuracy[i] += this_knn_accuracy  

        knn_f1[i] /= k
        knn_recall[i] /= k
        knn_precision[i] /=k
        knn_accuracy[i] /= k

    # Plotting statistics vs. values of k
    plt.scatter(k_nearest, knn_accuracy)
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.title("Value of K vs. KNN Accuracy")
    plt.savefig("./knn_accuracy.png")

    plt.clf()
    plt.scatter(k_nearest, knn_precision)
    plt.xlabel("K")
    plt.ylabel("Precision")
    plt.title("Value of K vs. KNN Precision")
    plt.savefig("./knn_precision.png")

    plt.clf()
    plt.scatter(k_nearest, knn_recall)
    plt.xlabel("K")
    plt.ylabel("Recall")
    plt.title("Value of K vs. KNN Recall")
    plt.savefig("./knn_recall.png")

    plt.clf()
    plt.scatter(k_nearest, knn_f1)
    plt.xlabel("K")
    plt.ylabel("F1 Score")
    plt.title("Value of K vs. KNN F1 Score")
    plt.savefig("./knn_f1.png")

    # Getting max scores
    max_accuracy  = np.argmax(knn_accuracy)
    max_precision = np.argmax(knn_precision)
    max_recall    = np.argmax(knn_recall)
    max_f1        = np.argmax(knn_f1)
    print("Max KNN Accuracy [%d-fold, k=%d]: %f" % (k, max_accuracy+1, knn_accuracy[max_accuracy]))
    print("Max KNN Precision [%d-fold, k=%d]: %f" % (k, max_precision+1, knn_precision[max_precision]))
    print("Max KNN Recall [%d-fold, k=%d]: %f" % (k, max_recall+1, knn_recall[max_recall]))
    print("Max KNN F1 [%d-fold, k=%d]: %f" % (k, max_f1+1, knn_f1[max_f1]))