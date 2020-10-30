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

"""
    Created by Connor Malley on 08/29/20

    CAP5106 HW1 Titanic Code
"""

seed(1)
plt.ion()

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
            train['Fare'][i] = fare_mode


    print('\n')
    print('Fare Mode == %f' % (fare_mode))    


    # %%
    # Question 20 

    # Setting fare to an ordinal value with described fare bands
    for i in range(len(train['Fare'])):
        if (train['Fare'][i] > -0.001 and train['Fare'][i] <= 7.91):
            train['Fare'][i] = 0
        elif (train['Fare'][i] > 7.91 and train['Fare'][i] <= 14.454):
            train['Fare'][i] = 1
        elif (train['Fare'][i] > 14.454 and train['Fare'][i] <= 31):
            train['Fare'][i] = 2
        elif (train['Fare'][i] > 31 and train['Fare'][i] <= 512.329):
            train['Fare'][i] = 3


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
