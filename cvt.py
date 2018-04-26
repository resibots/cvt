# Python code for creating a CVT
# Vassilis Vassiliades - Inria, Nancy - April 2018

import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
import argparse

# Default values
num_centroids = 7
dimensionality = 2
num_samples = 100000
num_replicates = 1
max_iterations = 100
tolerance = 0.0001
verbose = True

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dimensionality", type=int, help="space dimensionality (default: " + str(dimensionality) + ")")
parser.add_argument("-c", "--centroids", type=int, help="number of centroids (default: " + str(num_centroids) + ")")
parser.add_argument("-n", "--numsamples", type=int, help="number of sampled points (default: " + str(num_samples) + ")")
parser.add_argument("-i", "--iterations", type=int, help="maximum number of kmeans iterations (default: " + str(max_iterations) + ")")
parser.add_argument("-r", "--restarts", type=int, help="number of kmeans restarts (default: " + str(num_replicates) + ")")
parser.add_argument("-t", "--tolerance", type=float, help="tolerance level (default: " + str(tolerance) + ")")
parser.add_argument("-v", "--verbose", type=bool, help="increase output verbosity (default: " + str(verbose) + ")")
args = parser.parse_args()

# Change the parameters based on the arguments
if args.dimensionality:
    dimensionality = int(args.dimensionality)
if args.centroids:
    num_centroids = int(args.centroids)
if args.numsamples:
    num_samples = int(args.numsamples)
if args.restarts:
    num_replicates = int(args.restarts)
if args.iterations:
    max_iterations = int(args.iterations)
if args.tolerance:
    tolerance = float(args.tolerance)
if args.verbose:
    verbose = bool(args.verbose)

print 'Using:'
print 'num_centroids=',num_centroids
print 'dimensionality=',dimensionality
print 'num_samples=',num_samples
print 'num_replicates=',num_replicates
print 'max_iterations=',max_iterations
print 'tolerance=',tolerance
print 'verbose=',tolerance

X = np.random.rand(num_samples,dimensionality)

kmeans = KMeans(
    init='k-means++', 
    n_clusters=num_centroids, 
    n_init=num_replicates, 
    n_jobs=-1, 
    max_iter=max_iterations, 
    tol=tolerance,
    verbose=0)

kmeans.fit(X)
centroids = kmeans.cluster_centers_

filename = 'centroids_' + str(num_centroids) + '_' + str(dimensionality) + '.dat'

with open(filename, 'w') as f:
    print 'Writing to',filename,'...'
    for p in centroids:
        for item in p:
            f.write(str(item) + ' ')
        f.write('\n')
