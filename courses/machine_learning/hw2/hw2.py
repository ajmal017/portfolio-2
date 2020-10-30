# %%
import pandas as pd
from pandas import DataFrame as df
import numpy as np
import scipy
from collections import Counter
import statistics
from statistics import mode
import matplotlib.pyplot as plt
from numpy.random import seed
import random
from itertools import combinations

import sklearn
from sklearn import tree, ensemble
from sklearn import preprocessing

"""
    Created by Connor Malley on 08/29/20

    CAP5106 HW2 Titanic Code
"""

seed(1)

if __name__ == "__main__":
    # Reading input data
    train = pd.read_csv("./train.csv")
    test  = pd.read_csv("./test.csv")
    gender_submission = pd.read_csv("./gender_submission.csv")

    # Gettings nans
    contains_nan = {key : False for key in train.keys()}

    # Displaying which features have missing values in training
    for key in df.keys(train):
        isnull = train[key].isnull()
        for val in isnull:
            if (val):
                print(f"Warning: Missing %s data in train dataset" % (key))
                contains_nan[key] = True
                break
    

    # Displaying which features have missing values in testing
    for key in df.keys(test):
        isnull = test[key].isnull()
        for val in isnull:
            if (val):
                print(f"Warning: Missing %s data in test dataset" % (key))
                break


    # Gets stats on the numerical features
    numerical_features = {
        'PassengerId' : {},
        'Survived' : {},
        'Pclass' : {},
        'Age' : {},
        'SibSp' : {},
        'Parch' : {},
        'Fare' : {}
    }
    for feature in numerical_features.keys():
        train_feature = train[feature]
        if contains_nan[feature]:
            isnan_arr = np.isnan(train[feature])
            train_feature = train[feature][~isnan_arr]


        # Count
        numerical_features[feature]['count'] = np.count_nonzero(~np.isnan(train_feature))

        # Mean
        numerical_features[feature]['mean'] = np.mean(train_feature)

        # STDDEV
        numerical_features[feature]['std'] = np.std(train_feature)

        # Min
        numerical_features[feature]['min'] = np.min(train_feature)

        # Percentiles
        numerical_features[feature]['25%'] = np.percentile(train_feature, 25)        
        numerical_features[feature]['50%'] = np.percentile(train_feature, 50)        
        numerical_features[feature]['75%'] = np.percentile(train_feature, 75)        

        # Max
        numerical_features[feature]['max'] = np.max(train_feature)


    print('\n')
    numerical_features = df(data=numerical_features)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(numerical_features)


# %%
# Question 10

    # Finds if women or men are more likely to survive
    isFemale = np.zeros(len(train['Sex']))
    isMale = np.zeros(len(train['Sex']))
    for i in range(len(train['Sex'])):
        isFemale[i] = (train['Sex'][i] == 'female')
        isMale[i] = (train['Sex'][i] == 'male')

    
    female_survived = train['Survived'][isFemale]
    male_survived = train['Survived'][isMale]

    percent_female_survived = np.mean(female_survived)*100
    percent_male_survived = np.mean(male_survived)*100

    print('\n')
    print(f"%f percent of women survived." % percent_female_survived)
    print(f"%f percent of men survived." % percent_male_survived)
    if (percent_female_survived > percent_male_survived):
        print("Women are more likely to survive.")
    elif (percent_female_survived < percent_male_survived):
        print("Women are less likely to survive.")
    else:
        print("Women and men are just as likely to survive.")


    # %%
    # Question 8

    # Getting stats on categorical features
    categorical_features = {
        'Pclass' : {},
        'Sex' : {},
        'Embarked' : {}
    }
    for feature in categorical_features.keys():
        train_feature = train[feature]
        if contains_nan[feature]:
            isnan_arr = []
            isempty_arr = []
            if (isinstance(train_feature[0], str)):
                isempty_arr = (train_feature == '')
                train_feature = train[feature][~isempty_arr]
            else:
                isnan_arr = np.isnan(train[feature])
                train_feature = train[feature][~isnan_arr]

        # Count
        categorical_features[feature]['Count'] = len(train_feature)

        # Most frequent val
        val_counts = train_feature.value_counts(sort=True)
        categorical_features[feature]['Top'] = val_counts.keys()[0]
        categorical_features[feature]['Freq'] = val_counts[val_counts.keys()[0]]

        # Unique vals
        categorical_features[feature]['Unique'] = len(val_counts.keys())

    # %%
    # Question 9
    print('\n')
    categorical_features = df(categorical_features)
    print(categorical_features)

    corr = np.corrcoef(train['Pclass'], train['Survived'])
    print(corr)
    if (abs(corr[0][1]) < 0.5):
        print(f'\npclass and survived have a weak correlation of %f,\n\
therefore pclass won\'t be included in the features for this model' % corr[0][1])
    else:
        print(f'\npclass and survived have a strong correlation of %f,\n\
therefore pclass will be included in the features for this model' % corr[0][1])

    # %%
    # Question 11
    survived = ~np.isnan(train['Survived']) & train['Survived'] & ~ np.isnan(train['Age'])
    not_survived = ~np.isnan(train['Survived']) & ~train['Survived'] & ~np.isnan(train['Age']) 
    ages_survived = train['Age'][survived]
    ages_not_survived = train['Age'][not_survived]
    bins=np.linspace(0,max(train['Age']), num=int(max(train['Age']))+1, endpoint=True)
    fig, ax = plt.subplots(1, 2, sharey=True)
    ax[1].hist(ages_survived, bins=bins)
    ax[0].hist(ages_not_survived, bins=bins)
    ax[0].set_xlabel('Ages')
    ax[1].set_xlabel('Ages')
    ax[0].set_ylabel('Num survived')
    ax[1].set_title('Survived = 1')
    ax[0].set_title('Survived = 0')
    plt.savefig("hist1.png", dpi=300)

    # %%
    # Question 12

    # Comparing each Pclass with survived
    p1_s0 = (train['Pclass'] == 1) & (train['Survived'] == 0) & ~np.isnan(train['Age'])
    age_p1_s0 = train['Age'][p1_s0]
    p1_s1 = (train['Pclass'] == 1) & (train['Survived'] == 1) & ~np.isnan(train['Age'])
    age_p1_s1 = train['Age'][p1_s1]
    p2_s0 = (train['Pclass'] == 2) & (train['Survived'] == 0) & ~np.isnan(train['Age'])
    age_p2_s0 = train['Age'][p2_s0]
    p2_s1 = (train['Pclass'] == 2) & (train['Survived'] == 1) & ~np.isnan(train['Age'])
    age_p2_s1 = train['Age'][p2_s1]
    p3_s0 = (train['Pclass'] == 3) & (train['Survived'] == 0) & ~np.isnan(train['Age'])
    age_p3_s0 = train['Age'][p3_s0]
    p3_s1 = (train['Pclass'] == 3) & (train['Survived'] == 1) & ~np.isnan(train['Age'])
    age_p3_s1 = train['Age'][p3_s1]

    fig, ax = plt.subplots(3,2, sharey=True, sharex=True)
    ax[0][0].hist(age_p1_s0)
    ax[0][0].set_title('Pclass = 1 | Survived = 0')
    ax[0][1].hist(age_p1_s1)
    ax[0][1].set_title('Pclass = 1 | Survived = 1')
    ax[1][0].hist(age_p2_s0)
    ax[1][0].set_title('Pclass = 2 | Survived = 0')
    ax[1][1].hist(age_p2_s1)
    ax[1][1].set_title('Pclass = 2 | Survived = 1')
    ax[2][0].hist(age_p3_s0)
    ax[2][0].set_title('Pclass = 3 | Survived = 0')
    ax[2][1].hist(age_p3_s1)
    ax[2][1].set_title('Pclass = 3 | Survived = 1')

    ax[0][0].set_ylabel('Num Survived')
    ax[1][0].set_ylabel('Num Survived')
    ax[2][0].set_ylabel('Num Survived')
    ax[2][0].set_xlabel('Ages')
    ax[2][1].set_xlabel('Ages')

    fig.tight_layout()
    plt.savefig("hist2.png", dpi=300)

    # %%
    # Question 13


    fig, ax = plt.subplots(3,2, sharey=True, sharex=True)
    indices = {
        0 : (train['Embarked'] == 'S') & (train['Survived'] == 0) & (train['Sex'] == 'male'),
        1 : (train['Embarked'] == 'S') & (train['Survived'] == 0) & (train['Sex'] == 'female'),
        2 : (train['Embarked'] == 'S') & (train['Survived'] == 1) & (train['Sex'] == 'male'),
        3 : (train['Embarked'] == 'S') & (train['Survived'] == 1) & (train['Sex'] == 'female'),
        4 : (train['Embarked'] == 'C') & (train['Survived'] == 0) & (train['Sex'] == 'male'),
        5 : (train['Embarked'] == 'C') & (train['Survived'] == 0) & (train['Sex'] == 'female'),
        6 : (train['Embarked'] == 'C') & (train['Survived'] == 1) & (train['Sex'] == 'male'),
        7 : (train['Embarked'] == 'C') & (train['Survived'] == 1) & (train['Sex'] == 'female'),
        8 : (train['Embarked'] == 'Q') & (train['Survived'] == 0) & (train['Sex'] == 'male'),
        9 : (train['Embarked'] == 'Q') & (train['Survived'] == 0) & (train['Sex'] == 'female'),
        10 : (train['Embarked'] == 'Q') & (train['Survived'] == 1) & (train['Sex'] == 'male'),
        11 : (train['Embarked'] == 'Q') & (train['Survived'] == 1) & (train['Sex'] == 'female')
    }

    embarked = ['S', 'P', 'Q']
    survived = [0, 1]


    i = 0
    j = 0
    k = 0 
    for i in range(3):
        for j in range(2):
            data = {'female' : np.mean(train['Fare'][indices[k+1]]), 
            'male' : np.mean(train['Fare'][indices[k]])}
            ax[i][j].bar(data.keys(), data.values())
            ax[i][j].set_title(f'Embarked %s | Survived %d' % (embarked[i], survived[j]))
            k += 2

    ax[2][0].set_xlabel('Sex')    
    ax[2][1].set_xlabel('Sex')    
    ax[0][0].set_ylabel('Fare')
    ax[1][0].set_ylabel('Fare')
    ax[2][0].set_ylabel('Fare')
    fig.tight_layout()
    plt.savefig("hist3.png", dpi=300)


    # %%
    # Question 14
    duplicates = train['Ticket'].duplicated()
    count = 0
    for i in duplicates:
        if (i):
            count += 1

    print(f'\nThere are %d duplicate values for ticket, so %f percent' % (count, count/len(train['Ticket'])*100))


    ticket_vs_survived = {}
    i=0
    for ticket in train['Ticket']:
        if (ticket != '') :
            if not (ticket in ticket_vs_survived):
                ticket_vs_survived[ticket] = 0

            if (train['Survived'][i] == 1):
                ticket_vs_survived[ticket] += 1
        i += 1

    plt.cla()
    plt.clf()
    plt.close('all')
    plt.bar(ticket_vs_survived.keys(), ticket_vs_survived.values())
    plt.xlabel('Ticket')
    plt.ylabel('Num Survived')
    plt.title('Ticker vs. Survived')
    plt.savefig('hist4.png', dpi=300)

    # %%
    # Question 15

    cabin_isempty = ~(train['Cabin'] == '')
    print(f'\n%d passengers, but only %d entries for cabin' 
    % (len(train['Cabin']), train['Cabin'].count()))
    print('Therefore, the cabin feature won\'t be used.')


    # %%
    # Question 16

    gender = np.zeros(len(train['Sex']))
    for i in range(len(train['Sex'])):
        if (train['Sex'][i] == 'female'):
            gender[i] = 1.0
        elif (train['Sex'][i] == 'male'):
            gender[i] = 0.0
        else:
            gender[i] = float('NaN')

    # %%
    # Question 17


    # Method for replacing nan with val between mean and mean +- stddev
    ages_mean_std = train['Age']
    mean_age = np.mean(ages_mean_std)
    stddev_age = np.std(ages_mean_std)
    for age in ages_mean_std:
        if (np.isnan(age)):
            age = random.uniform(mean_age - stddev_age, mean_age + stddev_age) 




    # %%
    # Question 18

    # Replacing empty embarked vals with most common
    print('\n')
    isempty_arr = ~(train['Embarked'] == '')
    embarked_counts = train['Embarked'][isempty_arr].value_counts(sort=True)
    print(embarked_counts)

    for i in range(len(train['Embarked'])):
        if (train['Embarked'][i] == ''):
            train['Embarked'][i] = embarked_counts[0]


    # %%
    # Question 19

    # Setting nans in fare to the mode
    fare_mode = statistics.mode(train['Fare'])
    for i in range(len(train['Fare'])):
        if (np.isnan(train['Fare'][i])):
            train.loc[i, 'Fare'] = fare_mode


    print('\n')
    print('Fare Mode == %f' % (fare_mode))    


    # %%
    # Question 20 

    # Setting fare to an ordinal value with described fare bands
    for i in range(len(train['Fare'])):
        if (train['Fare'][i] > -0.001 and train['Fare'][i] <= 7.91):
            train.loc[i, 'Fare'] = 0
        elif (train['Fare'][i] > 7.91 and train['Fare'][i] <= 14.454):
            train.loc[i, 'Fare'] = 1
        elif (train['Fare'][i] > 14.454 and train['Fare'][i] <= 31):
            train.loc[i, 'Fare'] = 2
        elif (train['Fare'][i] > 31 and train['Fare'][i] <= 512.329):
            train.loc[i, 'Fare'] = 3


    # Printing percent survived vs. fare category
    percent0 = train['Fare'][train['Fare'] == 0].count() / len(train['Fare'])   
    percent1 = train['Fare'][train['Fare'] == 1].count() / len(train['Fare'])   
    percent2 = train['Fare'][train['Fare'] == 2].count() / len(train['Fare'])   
    percent3 = train['Fare'][train['Fare'] == 3].count() / len(train['Fare'])  
    print('\n') 
    print('Fare : 0, Survived: %f' % percent0)
    print('Fare : 1, Survived: %f' % percent1)
    print('Fare : 2, Survived: %f' % percent2)
    print('Fare : 3, Survived: %f' % percent3)


    #plt.show()
    #############################################################################################


    """ HOMEWORK 2"""

    mmScaler = preprocessing.MinMaxScaler()
    mmScaler.fit(np.array(train['Age']).reshape(-1, 1))

    # Normalizing age data and replacing NaNs with mean
    agesTrain = train['Age'].values.astype(float)
    agesTest  = test['Age'].values.astype(float)

    # Replacing NaNs with mean
    avgAgeTrain = np.mean(train['Age'])
    avgAgeTest = np.mean(test['Age'])
    for i in range(len(train['Age'])):
        if np.isnan(agesTrain[i]):
            agesTrain[i] = avgAgeTrain
    for i in range(len(test['Age'])):
        if np.isnan(agesTest[i]):
            agesTest[i] = avgAgeTest

    # Normalizing...
    train_preprocessed = pd.DataFrame()
    test_preprocessed = pd.DataFrame()
    train_preprocessed['Age'] = mmScaler.transform(np.array(agesTrain).reshape(-1, 1)).reshape(len(agesTrain))
    test_preprocessed['Age']  = mmScaler.transform(np.array(agesTest).reshape(-1, 1)).reshape(len(agesTest))
    continuousTrain = train_preprocessed['Age'].values.astype(float)
    continuousTest = test_preprocessed['Age'].values.astype(float)


    meanAge = np.floor(np.mean(continuousTrain))
    for i in range(len(continuousTrain)):
        if np.isnan(continuousTrain[i]):
            continuousTrain[i] = meanAge
    for i in range(len(continuousTest)):
        if np.isnan(continuousTest[i]):
            continuousTest[i] = meanAge

    # Encoding categorical data (nominal and ordinal)
    oneHotEnc = preprocessing.OneHotEncoder()
    labelEnc  = preprocessing.LabelEncoder()

    embarkedCounts = {
        'S' : 0,
        'P' : 0,
        'Q' : 0
    }

    embarkedCount = np.count_nonzero((train['Embarked'] == 'S') | (train['Embarked'] == 'P') | (train['Embarked'] == 'Q'))
    embarkedCounts['S'] = np.count_nonzero(train['Embarked'][train['Embarked'] == 'S'])
    embarkedCounts['P'] = np.count_nonzero(train['Embarked'][train['Embarked'] == 'P'])
    embarkedCounts['Q'] = np.count_nonzero(train['Embarked'][train['Embarked'] == 'Q'])
    embarkedCounts['S'] /= embarkedCount    
    embarkedCounts['P'] /= embarkedCount    
    embarkedCounts['Q'] /= embarkedCount    
    maxKey = max(embarkedCounts, key=embarkedCounts.get)
    print(f"MAX KEY: %s" % maxKey)


    # Replacing with most common value
    trainEmbarked = train['Embarked'].values
    trainEmbarked[pd.isnull(trainEmbarked)] = maxKey

    # Exporting to csv
    train.to_csv("../hw3/titanic_train.csv", index=False)
    test.to_csv("../hw3/titanic_test.csv", index=False)


    oneHotEnc.fit(np.array(train[['Sex', 'Embarked']]))
    labelEnc.fit(np.array(train['Pclass']))

    # Encoding the categorical data
    nominalTrain = oneHotEnc.transform(train[['Sex', 'Embarked']])
    ordinalTrain = labelEnc.transform(train['Pclass'])
    nominalTest = oneHotEnc.transform(test[['Sex', 'Embarked']])
    ordinalTest = labelEnc.transform(test['Pclass'])


    ordinalTrain = ordinalTrain.reshape((len(ordinalTrain), -1))
    ordinalTest  = ordinalTest.reshape((len(ordinalTest), -1))
    continuousTrain = continuousTrain.reshape((len(continuousTrain), -1))
    continuousTest  = continuousTest.reshape((len(continuousTest), -1))

    nominalTrain = nominalTrain.toarray()
    nominalTest = nominalTest.toarray()

    trainData = np.hstack((nominalTrain, ordinalTrain, continuousTrain))
    testData = np.hstack((nominalTest, ordinalTest, continuousTest))
    trainLabels = train['Survived'].values.astype(float)

    def createDecisionTree(features, labels, max_features=None):
        dTree = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_features=max_features)
        dTree.fit(features, labels)
        return dTree

    fig = plt.figure(figsize=(10,10))

    # Splitting data into training and testing sets
    percentTrain = 0.8
    samples = len(trainData)
    shuffler = np.random.permutation(samples)
    trainData = trainData[shuffler]
    trainLabels = trainLabels[shuffler]
    validationData = trainData[int(percentTrain*samples)::]
    validationLabels = trainLabels[int(percentTrain*samples)::]

    trainData = trainData[0:int(percentTrain*samples):]
    trainLabels = trainLabels[0:int(percentTrain*samples):]


    dTree = createDecisionTree(trainData, trainLabels)
    tree.plot_tree(dTree, max_depth=5, fontsize=10)

    predictions = dTree.predict(validationData)
    def get_accuracy(predictions, labels):
        n = len(predictions)
        n_correct = 0
        for i in range(len(predictions)):
            if predictions[i] == labels[i]:
                n_correct += 1
        
        return n_correct / n


    # Shuffles features and labels and returns k unique splits
    def get_k_datasets(features, labels, k=1):
        shuffler = np.random.permutation(len(features))
        features_shuffled = features[shuffler]
        labels_shuffled = labels[shuffler]

        split_size = len(features) // k

        feature_splits = []
        label_splits   = []

        startInd = 0
        for i in range(k):
            stopInd = (i+1)*split_size
            if (i == (k-1)):
                stopInd = len(features)

            feature_splits.append(features_shuffled[startInd:stopInd])
            label_splits.append(labels_shuffled[startInd:stopInd])

            startInd = stopInd

        return feature_splits, label_splits

    
    print(f"Decision tree accuracy (no cross validation) : %f" % get_accuracy(predictions, trainLabels))
    fig.tight_layout()


    # Creates K models
    k = 5
    trainData = np.hstack((nominalTrain, ordinalTrain, continuousTrain))
    testData = np.hstack((nominalTest, ordinalTest, continuousTest))
    trainLabels = train['Survived'].values.astype(float)

    print(f"Splitting into %d datasets" % k)
    feature_splits, label_splits = get_k_datasets(trainData, trainLabels, k=k)
    models = [tree.DecisionTreeClassifier(criterion='gini', splitter='best') for i in range(len(feature_splits))]
    for i in range(len(models)):
        featureInds = np.arange(k)
        featureInds = np.delete(featureInds, np.where(featureInds == i))
        train_features = feature_splits[featureInds[0]]
        train_labels   = label_splits[featureInds[0]]
        for ind in featureInds[1:]:
            train_features = np.concatenate([train_features, feature_splits[ind]], axis=0) 
            train_labels   = np.concatenate([train_labels, label_splits[ind]], axis=0)


        models[i].fit(train_features, train_labels)

    # Getting prediction accuracy    
    accuracy = np.zeros(k) 
    for i in range(len(models)):
        test_features = feature_splits[i] 
        test_labels   = label_splits[i]

        # Predicting on one of datasets not trained on
        predictions = models[i].predict(test_features)
        accuracy[i] = get_accuracy(predictions, test_labels)
        print(f"Accuracy of split # %d == %f" % (i, accuracy[i]))

    print(f"Average accuracy across %d splits == %f" % (k, np.mean(accuracy)))

    # Random forest algorithm
    m = 5
    trainData = np.hstack((nominalTrain, ordinalTrain, continuousTrain))
    testData = np.hstack((nominalTest, ordinalTest, continuousTest))
    random_forest = ensemble.RandomForestClassifier(n_estimators=200, min_samples_split=10, 
    oob_score=True, random_state=1)

    # Splitting data into training and testing sets
    percentTrain = 0.8
    samples = len(trainData)
    shuffler = np.random.permutation(samples)
    trainData = trainData[shuffler]
    trainLabels = trainLabels[shuffler]
    validationData = trainData[int(percentTrain*samples)::]
    validationLabels = trainLabels[int(percentTrain*samples)::]

    trainData = trainData[0:int(percentTrain*samples):]
    trainLabels = trainLabels[0:int(percentTrain*samples):]

    random_forest.fit(trainData, trainLabels)
    predictions = random_forest.predict(validationData)

    print(f"Random forest accuracy : %f" % get_accuracy(predictions, trainLabels))











#######################################################
# Ignore, this was my own attempt at creating a decision tree from scratch
"""



def gini(df, key='Survived'):

    # Corresponds to yes/no counts for survived 
    val_counts = {
        'yes' : 0,  
        'no' : 0}

    val_counts['yes'] = np.count_nonzero(df[key] == 'yes')
    val_counts['no'] = np.count_nonzero(df[key] == 'no')
    n = len(df[key])

    gini = 1 - (val_counts['yes']/n)**2 - (val_counts['no']/n)**2
    return gini

def splitContinuous(feature, df, key='Survived'):
    sorted_feature = list(df[feature]).sort()

    # Possible values used for comparison
    comp_vals     = np.zeros(len(sorted_feature)+1)
    comp_vals[0]  = sorted_feature[0]  - (sorted_feature[1] - sorted_feature[0])   / 2
    comp_vals[-1] = sorted_feature[-1] + (sorted_feature[-1] - sorted_feature[-2]) / 2 
    for i in range(1, len(sorted_features)-1):
        comp_vals[i] = sorted_feature[i-1] + (sorted_feature[i] - sorted_feature[i-1]) / 2

    gini_vals = np.zeros(len(com_vals))

    # Collecting gini vals of all possible binary splits to find optimal split
    yes_lt_count = np.zeros(len(comp_vals))
    yes_gt_count = np.zeros(len(comp_vals))
    no_lt_count  = np.zeros(len(comp_vals))
    no_gt_count  = np.zeros(len(comp_vals))
    for i in range(1, len(comp_vals)):
        yes_lt_count[i] += int(df['Survived'][i-1] == 'yes')
        no_lt_count[i]  += int(df['Survived'][i-1] == 'no')

    for i in reversed(range(len(comp_vals)-1)):
        yes_gt_count[i] += int(df['Survived'][i+1] == 'yes' and df['Survived'][i+1] > comp_vals[i])
        no_gt_count[i]  += int(df['Survived'][i+1] == 'no' and df['Survived'][i+1] > comp_vals[i])

    for i  in range(len(comp_vals)):
        n_lt = yes_lt_count[i] + no_lt_count[i]
        n_gt = yes_gt_count[i] + no_gt_count[i]
        n = n_lt + n_gt
        
        gini_lt = 1 - (yes_lt_count[i]/n_lt)**2 - (no_lt_count[i]/n_lt)**2
        gini_gt = 1 - (yes_lt_count[i]/n_lt)**2 - (no_lt_count[i]/n_lt)**2
        gini_vals[i] = (n_lt/n) * gini_lt + (n_gt/n) * gini_gt 

    min_gini_ind = gini_vals.index(min(gini_vals))

    return gini_vals[min_gini_ind], comp_vals[min_gini_ind]

def splitNominal(feature, df, key='Survived'):
    categories = np.unique(df[feature])
    vals = {}
    for val in categories:
        vals[val] = {
            'inds' : df[feature] == val,
            }
        vals[val]['n'] = np.count_nonzero(inds)
        vals[val]['yes_count'] = np.count_nonzero(df[feature][vals[val][inds]] == 'yes')
        vals[val]['no_count'] = np.count_nonzero(df[feature][vals[val][inds]] == 'no')


    num_vals = len(categories)
    split_combos = []

    # Getting all possible combinations to be used in binary split
    for i in range(1,np.ceil(len(split_combos)/2)):
        split_combos.append(combinations(categories, i))

    gini_vals = np.zeros(split_combos)


    for i, combo in enumerate(split_combos, start=0):
        left_yes_count = 0
        left_no_count  = 0
        left_n   = 0
        right_yes_count = 0
        right_no_count  = 0
        right_n   = 0
        for val in categories:
            if val in combo:
                left_yes_count += vals[val]['yes_count']
                left_no_count  += vals[val]['no_count']
                left_n  += vals[val]['n']
            else:
                right_yes_count += vals[val]['yes_count']
                right_no_count  += vals[val]['no_count']
                right_n  += vals[val]['n']

        # TODO: Do same calculation for those not in combo
        gini_val_left = 1 - (left_yes_count/left_n)**2 - (left_no_count/left_n)**2
        gini_val_right = 1 - (right_yes_count/right_n)**2 - (right_no_count/right_n)**2
        gini_vals[i] = (left_n/(left_n+right_n))*gini_val_left + (right_n/(left_n+right_n))*gini_val_right

    min_gini_val = np.min(gini_vals)
    min_gini_ind = gini_vals.index(min_gini_val)
    return min_gini_val, split_combos[min_gini_ind]


    

def initTree(df):
    # Checks if > 1 class exists in current rows
    isSameClass =  not np.any(df['Survived'][df['Survived'] != df['Survived'][0]])

    # Case for leaf node
    if isSameClass:
        return df['Survived'][0]
    
    gini_prev = gini(df)

    # If this is reached, data must be split again
    gini_split = 0
    n = len(df['Survived'])

    wasAdded = {}
    for feature in df.to_dict().keys():
        wasAdded[feature] = False

    for i in range(len(df.to_dict().keys())):
        for feature in df.to_dict().keys():
            # TODO: Have user input type of data (or auto-detect)
            if feature == 'Age' and not wasAdded[feature]:
                min_age_gini, age_split_val = splitContinuous('Age', df)
                continue

            # Just calculating split in place since only 2 possible values for sex
            if feature == 'Sex' and not wasAdded[feature]:
                min_sex_gini, sex_split_vals = splitNominal('Sex', df)
                continue

            if feature == 'Embarked' and not wasAdded[feature]:
                min_embarked_gini, embarked_split_vals = splitNominal('Embarked', df)
                continue
        

"""

# %%
