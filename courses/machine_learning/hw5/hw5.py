import numpy as np
import sklearn
from sklearn import datasets, preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import statistics
import sys
import random

"""
Connor Malley
CAP5610
HW5
"""
# Gets euclidean distance between 2 points
def euclidean_distance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i])**2

    return np.sqrt(distance)

# Gets Manhattan distance between 2 points
def manhattan_distance(x1, x2):
    distance = 0
    if len(x1) != len(x2):
        return None

    for i in range(len(x1)):
        distance += np.abs(x1[i] - x2[i])

    return distance

# Gets jaccard distance between 2 points
def jaccard_distance(x1, x2):
    if len(x1) !=  len(x2):
        return None

    min_term = 0
    max_term = 0
    for i in range(len(x1)):
        min_term += np.min([x1[i], x2[i]])
        max_term += np.max([x1[i], x2[i]])

    return 1 - (min_term/max_term)

# Gets Cosine distance between 2 points
def cosine_distance(x1, x2):
    if len(x1) != len(x2):
        return None

    # Similarity
    x1_mag = np.sqrt(np.sum(np.multiply(x1, x1)))
    x2_mag = np.sqrt(np.sum(np.multiply(x2, x2)))
    denom = x1_mag * x2_mag
    numer = np.sum(np.multiply(x1, x2))
    if denom == 0:
        return 1

    return 1 - (numer/denom)

# Gets SSE of clustering
def sse(X, clustering, centroids, distance=manhattan_distance):
    sse_sum = 0
    for i in range(len(X)):
        sse_sum += distance(X.loc[i], centroids[clustering[i]])**2

    return sse_sum

# Gets the position of the centroid of some dataset (mean)
def get_centroid(X, centroid='mean'):
    return np.mean(X, axis=0)


# Gets the clustering for a set of samples and centroids
def get_clustering(X, centroids, distance=manhattan_distance):
    clustering = np.zeros(len(X), dtype=np.int32)
    cluster_distances = np.zeros(len(centroids))
    for i in range(len(X)):
        for j in range(len(centroids)):
           cluster_distances[j] = distance(X.iloc[i].values, centroids[j])
        
        clustering[i] = np.argmin(cluster_distances)

    return clustering

# Updates the centroids based on the given clustering
def move_centroids(X, clustering, centroids):
    new_centroids = np.zeros(np.shape(centroids))
    for i in range(len(centroids)):
        cluster_mean = np.mean(X.iloc[clustering == i], axis=0).values

        # If no data points in a given cluster, don't move that centroid
        if np.isnan(cluster_mean[0]):
            new_centroids[i] = centroids[i]
        else:
            new_centroids[i] = cluster_mean


    return new_centroids
            
# Finds the K-Means clustering of the input dataset
def kmeans_clustering(X, k, centroids, distance=manhattan_distance, iterations=None, 
terminator='centroid', print_iters=False):
    new_centroids = np.copy(centroids)
    old_clustering = np.zeros(len(X))

    if iterations == None:
        iterations = sys.maxsize

    iters = 0
    old_sse = sys.maxsize
    new_sse = sys.maxsize
    for i in range(iterations):
        old_centroids = np.copy(new_centroids)
        old_clustering = get_clustering(X, old_centroids, distance=distance)
        new_centroids = move_centroids(X, old_clustering, old_centroids)
        new_clustering = get_clustering(X, new_centroids, distance=distance)

        old_sse = sse(X, old_clustering, old_centroids, distance=distance) 
        new_sse = sse(X, new_clustering, new_centroids, distance=distance) 
        iters += 1
        if terminator == 'clustering':
            if np.array_equal(new_clustering, old_clustering):
                if (print_iters):
                    print('Iterations: %d' % iters)
                return old_clustering, new_centroids
        elif terminator == 'centroid':
            if np.array_equal(new_centroids, old_centroids):
                if (print_iters):
                    print('Iterations: %d' % iters)
                return old_clustering, new_centroids
        elif terminator == 'sse':
            if new_sse >= old_sse:
                if (print_iters):
                    print('Iterations: %d' % iters)
                return old_clustering, new_centroids


    if (print_iters):
        print('Iterations: %d' % iters)

    return new_clustering, new_centroids


if __name__ == '__main__':
    task1 = True
    task2 = True
    task4 = True

    if task1:

        # Question 1
        print('Question 1\n==============')
        football_data = pd.DataFrame(index=np.arange(0, 10))
        wins_2016 = [3, 3, 2, 2, 6, 6, 7, 7, 8, 7]
        wins_2017 = [5, 4, 8, 3, 2, 4, 3, 4, 5, 6]
        football_data['wins_2016'] = wins_2016
        football_data['wins_2017'] = wins_2017

        # Original Dataset
        def create_football_plot():
            plt.scatter(wins_2016, wins_2017)
            plt.xlabel('# Wins in 2016')
            plt.ylabel('# Wins in 2017')
            plt.title('# Wins in 2016 vs. 2017')

        create_football_plot()
        plt.savefig('football_data_plot.png', dpi=300)
        plt.clf()

        def create_football_kmeans_plot(X, clustering=[], centroids=[], fig_path='plot.png', title='# Wins 2016 vs. 2017'):
            colors = ['blue', 'red']

            if len(clustering) > 0:
                clusters = list(set(clustering))
                for i in range(len(clusters)):
                    plt.scatter(X.iloc[clustering == i].values[:,0], X.iloc[clustering == i].values[:,1],
                    color=colors[i], label='Cluster %d' % (i))
            else:
                plt.scatter(X[X.columns[0]], X[X.columns[1]],
                color=colors[0])


            plt.scatter(centroids[:,0], centroids[:,1], c='black', marker='x', s=128, label='Centroid')
            plt.xlabel('# Wins in 2016')
            plt.ylabel('# Wins in 2017')
            plt.title(title)
            plt.legend()
            plt.savefig(fig_path, dpi=300)
            plt.clf()



        # Trial 1 - (Manhattan Distance)
        centroids = np.array([[4, 6], [5, 4]])
        create_football_kmeans_plot(football_data, [], centroids, 'trial_1_0.png', 'K-Means Initialization') # Init
        clustering, centroids = kmeans_clustering(football_data, 2, centroids, distance=manhattan_distance, iterations=1) # Iter 1
        print('trial 1 first iter centroids:\n%s' % (str(centroids)))
        create_football_kmeans_plot(football_data, clustering, centroids, 'trial_1_1.png', 'K-Means Clustering 1st iteration')
        clustering, centroids = kmeans_clustering(football_data, 2, centroids, distance=manhattan_distance, iterations=None, terminator='centroid') # Final Iter
        create_football_kmeans_plot(football_data, clustering, centroids, 'trial_1_final.png', 'K-Means Clustering Final')

        # Trial 2 - (Euclidean Distance)
        centroids = np.array([[4, 6], [5, 4]])
        create_football_kmeans_plot(football_data, [], centroids, 'trial_2_0.png', 'K-Means Initialization') # Init
        clustering, centroids = kmeans_clustering(football_data, 2, centroids, distance=euclidean_distance, iterations=1) # Iter 1
        print('trial 2 first iter centroids:\n%s' % (str(centroids)))
        create_football_kmeans_plot(football_data, clustering, centroids, 'trial_2_1.png', 'K-Means Clustering 1st iteration')
        clustering, centroids = kmeans_clustering(football_data, 2, centroids, distance=euclidean_distance, iterations=None, terminator='centroid') # Final Iter
        create_football_kmeans_plot(football_data, clustering, centroids, 'trial_2_final.png', 'K-Means Clustering Final')

        # Trial 3 - (Manhattan Distance)
        centroids = np.array([[3, 3], [8, 3]])
        create_football_kmeans_plot(football_data, [], centroids, 'trial_3_0.png', 'K-Means Initialization') # Init
        clustering, centroids = kmeans_clustering(football_data, 2, centroids, distance=manhattan_distance, iterations=1) # Iter 1
        print('trial 3 first iter centroids:\n%s' % (str(centroids)))
        create_football_kmeans_plot(football_data, clustering, centroids, 'trial_3_1.png', 'K-Means Clustering 1st iteration')
        clustering, centroids = kmeans_clustering(football_data, 2, centroids, distance=manhattan_distance, iterations=None, terminator='centroid') # Final Iter
        create_football_kmeans_plot(football_data, clustering, centroids, 'trial_3_final.png', 'K-Means Clustering Final')

        # Trial 4 - (Manhattan Distance)
        centroids = np.array([[3, 2], [4, 8]])
        create_football_kmeans_plot(football_data, [], centroids, 'trial_4_0.png', 'K-Means Initialization') # Init
        clustering, centroids = kmeans_clustering(football_data, 2, centroids, distance=manhattan_distance, iterations=1) # Iter 1
        print('trial 4 first iter centroids:\n%s' % (str(centroids)))
        create_football_kmeans_plot(football_data, clustering, centroids, 'trial_4_1.png', 'K-Means Clustering 1st iteration')
        clustering, centroids = kmeans_clustering(football_data, 2, centroids, distance=manhattan_distance, iterations=None) # Final Iter
        create_football_kmeans_plot(football_data, clustering, centroids, 'trial_4_final.png', 'K-Means Clustering Final')


    # Task 2
    if task2:
        print('Question 2\n=============')
        # Load Dataset
        iris = datasets.load_iris(as_frame=True)
        X = iris.data  # we only take the first two features.
        Y = iris.target
        names = iris.target_names

        # For initialization
        random.seed(1)

        # Normalize features
        mmScaler = preprocessing.MinMaxScaler()
        X.loc[:][:] = mmScaler.fit_transform(np.array(X.values))
        #Y = mmScaler.fit_transform(Y.values.reshape(-1, 1))

        # Q1
        k = 3
        centroids = np.zeros((3, 4))
        for i in range(len(centroids[:,0])):
            for j in range(len(centroids[0,:])):
                centroids[i, j] = random.uniform(0, 1)

        euc_clustering, euc_centroids = kmeans_clustering(X, 3, centroids, euclidean_distance, print_iters=True)
        cos_clustering, cos_centroids = kmeans_clustering(X, 3, centroids, cosine_distance, print_iters=True)
        jar_clustering, jar_centroids = kmeans_clustering(X, 3, centroids, jaccard_distance, print_iters=True)
        
        # Using euclidean distance to compare the performance of the 3
        # (Initially I used the respective distances, but that just wouldn't make sense
        # as the cosine distance would not be proportional to the other 2)
        euc_sse = sse(X, euc_clustering, euc_centroids, euclidean_distance)
        cos_sse = sse(X, cos_clustering, cos_centroids, euclidean_distance)
        jar_sse = sse(X, jar_clustering, jar_centroids, euclidean_distance)
        print('Euclidean SSE: %f\nCosine SSE: %f\nJaccard SSE: %f\n' % (euc_sse, cos_sse, jar_sse))

        # Q2
        def get_clustering_accuracy(clustering, cluster_values, Y, name):
            clustering_accuracy = 0
            for i in range(len(cluster_values)):
                cluster = Y[clustering == cluster_values[i]]
                cluster_labels = list(set(cluster))

                
                # Initializing label votes
                label_votes = np.zeros(len(cluster_values))
                for val in cluster:
                    label_votes[val] += 1

                max_cluster = np.argmax(label_votes)
                max_votes  = np.max(label_votes)
                clustering_accuracy += (max_votes/len(clustering))

            print('Accuracy of %s: %f' % (name, clustering_accuracy))

        cluster_values = np.arange(0, 3)
        get_clustering_accuracy(euc_clustering, cluster_values, Y, 'Euclidean')
        get_clustering_accuracy(cos_clustering, cluster_values, Y, 'Cosine')
        get_clustering_accuracy(jar_clustering, cluster_values, Y, 'Jaccard')
        print('')

        # Q4
        #a
        euc_clustering, euc_centroids = kmeans_clustering(X, 3, centroids, euclidean_distance, terminator='centroid', print_iters=True)
        cos_clustering, cos_centroids = kmeans_clustering(X, 3, centroids, cosine_distance, terminator='centroid', print_iters=True)
        jar_clustering, jar_centroids = kmeans_clustering(X, 3, centroids, jaccard_distance, terminator='centroid', print_iters=True)
        euc_sse = sse(X, euc_clustering, euc_centroids, euclidean_distance)
        cos_sse = sse(X, cos_clustering, cos_centroids, euclidean_distance)
        jar_sse = sse(X, jar_clustering, jar_centroids, euclidean_distance)
        print('Termination on no centroid change\n===============\nEuclidean SSE: %f\nCosine SSE: %f\nJaccard SSE: %f\n' % (euc_sse, cos_sse, jar_sse))
         
        #b
        euc_clustering, euc_centroids = kmeans_clustering(X, 3, centroids, euclidean_distance, terminator='sse', print_iters=True)
        cos_clustering, cos_centroids = kmeans_clustering(X, 3, centroids, cosine_distance, terminator='sse', print_iters=True)
        jar_clustering, jar_centroids = kmeans_clustering(X, 3, centroids, jaccard_distance, terminator='sse', print_iters=True)
        euc_sse = sse(X, euc_clustering, euc_centroids, euclidean_distance)
        cos_sse = sse(X, cos_clustering, cos_centroids, euclidean_distance)
        jar_sse = sse(X, jar_clustering, jar_centroids, euclidean_distance)
        print('Termination on SSE increase\n===============\nEuclidean SSE: %f\nCosine SSE: %f\nJaccard SSE: %f\n' % (euc_sse, cos_sse, jar_sse))

        #c
        euc_clustering, euc_centroids = kmeans_clustering(X, 3, centroids, euclidean_distance, iterations=100, terminator='iterations', print_iters=True)
        cos_clustering, cos_centroids = kmeans_clustering(X, 3, centroids, cosine_distance, iterations=100, terminator='iterations', print_iters=True)
        jar_clustering, jar_centroids = kmeans_clustering(X, 3, centroids, jaccard_distance, iterations=100, terminator='iterations', print_iters=True)
        euc_sse = sse(X, euc_clustering, euc_centroids, euclidean_distance)
        cos_sse = sse(X, cos_clustering, cos_centroids, euclidean_distance)
        jar_sse = sse(X, jar_clustering, jar_centroids, euclidean_distance)
        print('Termination on 100 iterations\n===============\nEuclidean SSE: %f\nCosine SSE: %f\nJaccard SSE: %f\n' % (euc_sse, cos_sse, jar_sse))


    if (task4):
        # Task 4
        print('Question 4\n=============')
        class0 = [[4.7, 3.2], [4.9, 3.1], [5.0, 3.0], [4.6, 2.9]]
        class1 = [[5.9, 3.2], [6.7, 3.1], [6.0, 3.0], [6.2, 2.8]]
        max_distance = 0
        points= [[0, 0], [0, 0]] 
        for point0 in class0:
            for point1 in class1:
                if euclidean_distance(point0, point1) > max_distance:
                    max_distance = euclidean_distance(point0, point1)
                    points[0] = point0
                    points[1] = point1

        print('Points with max distance\n%s' % (str(points)))
        print('Max distance: %.4f\n' % (max_distance))

        min_distance = 999999
        points= [[0, 0], [0, 0]]
        for point0 in class0:
            for point1 in class1:
                if euclidean_distance(point0, point1) < min_distance:
                    min_distance = euclidean_distance(point0, point1)
                    points[0] = point0
                    points[1] = point1

        print('\nPoints with min distance\n%s' % (str(points)))
        print('Min distance: %.4f\n' % (min_distance))
        
        avg_distance = 0 
        points= [[0, 0], [0, 0]] 
        for point0 in class0:
            for point1 in class1:
                avg_distance += euclidean_distance(point0, point1)

        avg_distance /= len(class0)**2
        print('\n(Max + Min) / 2: %.4f' % ((max_distance+min_distance)/2))
        print('Average distance between pairs: %.4f' % (avg_distance))

        # Avg intra cluster distances
        avg_distance0 = 0 
        points = [[0, 0], [0, 0]]
        for point0 in class0:
            for point1 in class0:
                avg_distance0 += euclidean_distance(point0, point1)

        avg_distance0 /= len(class0)**2

        # Avg intra cluster distances
        avg_distance1 = 0 
        points = [[0, 0], [0, 0]]
        for point0 in class1:
            for point1 in class1:
                avg_distance1 += euclidean_distance(point0, point1)

        avg_distance1 /= len(class0)**2

        avg_distance = (avg_distance0 + avg_distance1) /2
        print('Average intra-cluster distance: %.4f' % (avg_distance))
