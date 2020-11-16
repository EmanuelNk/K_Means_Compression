import sys
import numpy as np
from numpy.core.fromnumeric import shape
import scipy.io.wavfile
from scipy.spatial import distance
import matplotlib.pyplot as plt



sample,centroids = sys.argv[1],sys.argv[2]
fs, y = scipy.io.wavfile.read(sample)
x=np.array(y.copy())
centroids = np.loadtxt(centroids)

#returns for each point, the closest centroid from the centroids array
def findClosestCentroid(points,centroidsArr):
    return (np.argmin(distance.cdist(points, centroidsArr),axis=1))

#Groups each point it's closest cluster
def groupToClusters(points, closestCentroids):
    clusters = []
    k = len(centroids)
    for x in range(k):
        clusters.append([])
    for i, centroid in enumerate(closestCentroids):
        clusters[centroid].append(points[i])
    return clusters

#calculates the new centroids by calculating the average point of each cluster   
def modifyCentroids(clusters):
    modifiedCentroids = []
    for cluster in clusters:
        modifiedCentroids.append(np.round(np.average(cluster, axis=0)))
    return (np.array(modifiedCentroids))

#for each cluster replace all values by it's centroid
def replaceByCentroids(data,centroids):
    closestCentroids = findClosestCentroid(data,centroids)
    newData = []
    for i in range(len(data)):
       newData.append(centroids[closestCentroids[i]])
    newData = np.array(newData, dtype=np.int16)
    return newData

#calculate the average cost of a given centroids list
def cost(points, centroids):
    costT = 0
    closestCent = findClosestCentroid(points,centroids)
    for i,point in enumerate(points):
        difference = point - centroids[closestCent[i]]
        magnitude = np.linalg.norm(difference)
        costT = costT + pow(magnitude, 2)
    return costT/len(points)

k= len(centroids)
prevCentroids = []
costsArr = []
costPrevious = 0
centroids_file = open("output.txt", "w")
iterations = 0
for j in range(30):
    closestCentroids = findClosestCentroid(y,centroids)
    clusters = groupToClusters(y,closestCentroids)
    centroids = modifyCentroids(clusters)
    centroids_file.write(f"[iter {j}]:{','.join([str(i) for i in centroids])}\n")
    costCurrent = cost(y,centroids)
    costsArr.append(costCurrent)
    iterations = iterations+1
    if costCurrent == costPrevious:
        break
    else:   
        prevCentroids = centroids
        costPrevious = costCurrent
centroids_file.close()


# to plot to a graph use the following code: 

"""
costsArr = np.array(costsArr)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot([x for x in range(1,iterations+1)], costsArr)
headline = ("K={}".format(k))
fig.suptitle(headline, fontsize=14, fontweight='bold')
ax.set_xlabel('Iteration')
ax.set_ylabel('Average Cost')
filename = ("K_{}_graph.png".format(k)) 
plt.savefig(filename)
plt.show()

"""

#to write a compressed file use the following code:

"""

compressedFile = replaceByCentroids(y,centroids)
scipy.io.wavfile.write("compressed.wav", fs, compressedFile)

"""

