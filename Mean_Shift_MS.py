import math
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import sklearn
from sklearn.cluster import estimate_bandwidth


pylab.rcParams['figure.figsize'] = 6, 6

# Data cluster  = X_tab
n_samples = 100     # number of samples

centers = [[1, 1], [-1, -1], [1, -1]]
# centers = 3
X_tab, Y_tab = make_blobs(n_samples=100, n_features=2, centers=centers, cluster_std=1.9)
print(X_tab.shape)
plt.plot(X_tab[:, 0], X_tab[:, 1], 'bo', markersize=10)
# plt.show() # shows graph of original data cluster

# Sklearn method of sigma estimation
# a = sklearn.cluster.estimate_bandwidth(X_tab, quantile=0.2, n_samples=500, random_state=0, n_jobs=None)
# sigma = a


def eucl_dist(qi,pi):    # Euclidean Distance
    eudist = np.sqrt(np.sum((qi-pi)**2))
    return eudist


def neigh_points(X, x, dist = 5):  # Distance from neighbouring points
    Y = []
    for i in X:
        d = eucl_dist(i, x)
        if d <= dist:
            Y.append(i)
    return Y


def gaussian_krnl(x, sigma):     # Gaussian Kernel
    ret = (1/(sigma*math.sqrt(2*math.pi))) * np.exp(-0.5*(x / sigma)**2)
    return ret


search_distance = 6 # Mean Shift parameter 1 / Diameter of the search circle
sigma = estimate_bandwidth(X_tab, quantile=0.2, n_samples=500)  # Standard deviation from gaussian kernel


X = np.copy(X_tab)  # X is an array copy of original points / X_tab
Y = []
n = 5    # Number of iterations

for k in range(n):
    for i, x in enumerate(X):
        # Find closest points for each point
        points = neigh_points(X, x, search_distance)

        # Calculate Mean Shift / m_of_x
        numr = 0    # numerator
        denomr = 0   # denominator
        for xi in points:
            distance = eucl_dist(xi, x)
            weight = gaussian_krnl(distance, sigma)
            numr += (weight * xi)
            denomr += weight
        m_of_x = numr / denomr
        # update table
        X[i] = m_of_x
    # append X to new table
    Y.append(np.copy(X))

# Plot font
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 12,
        }

# Plot the outputs
figure = plt.figure(1)
figure.set_size_inches((12, 9))
plt.subplot(n + 2, 1, 1)
plt.title(label='Original', fontdict=font)
plt.plot(X_tab[:,0], X_tab[:,1], 'o')

# Numerical coordinates of the first element of the last iteration / for debugging
# print(Y[n_iterations-1][0,0],Y[n_iterations-1][0,1])

for i in range(n):
    index = i + 2
    plt.subplot(n + 2, 1, index)
    plt.title(label='Iteration: %d' % (index - 1), fontdict=font)
    plt.plot(X_tab[:,0], X_tab[:,1], 'o')
    plt.plot(Y[i][:,0], Y[i][:,1], 'ro')
plt.show()

plt.plot(X_tab[:,0], X_tab[:,1], 'o')
plt.plot(Y[i][:,0], Y[i][:,1], 'ro')
print("centroid iteration",Y[n-1][0,0],Y[n-1][0,1])
plt.show()

###################################################################################
def sklearn_mean_shift(X_tab):   #sklearn faster method using developed functions
    import numpy as np
    from sklearn.cluster import MeanShift, estimate_bandwidth
    from sklearn.datasets.samples_generator import make_blobs

    # Generate sample data
    # centers = [[1, 1], [-1, -1], [1, -1]]
    # X, _ = make_blobs(n_samples=100, n_features= 2 , centers=centers, cluster_std=1.9)

    # Compute clustering with MeanShift

    # The following bandwidth can be automatically detected using
    # bandwidth = estimate_bandwidth(X_tab, quantile=0.2, n_samples=500)

    ms = MeanShift(bandwidth=100, bin_seeding=True)
    ms.fit(X_tab)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    # print("number of estimated clusters : %d" % n_clusters_)

    # Plot result
    # import matplotlib.pyplot as plt
    from itertools import cycle

    plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X_tab[my_members, 0], X_tab[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    print("centroid sklearn", cluster_center[0], cluster_center[1])
    a = cluster_center[0]
    b = cluster_center[1]
    return a, b     # returns final coordinates of sklearn cluster center


# sklearn_mean_shift(X_tab)    #uncoment to run better version of mean shift
a, b = sklearn_mean_shift(X_tab)    # coordinates of sklearn cluster center


# Errors between two methods
c = Y[n-1][0, 0]                        # x coordinate of iteration cluster center
d = Y[n-1][0, 1]                        # y coordinate of iteration cluster center
error = [abs(a-c),abs(b-d)]             # error table
print(" x_err: ",error[0],"\n", "y_err: ", error[1])

print("done")