from collections import defaultdict
import math
import matplotlib.pyplot as plt


data = [(6.,20.), (8.,15.), (3.,4.), (1.,1.), (21.,19.),(5.,4.),(25.,29.)] #
manhattan = False # decide to use manhattan instead of euclidean distance
k = 2 # number of clusters
centers = [(2.1,1.0),(2.1,2.0)] # Cluster centers

# Distance between 2 points
def distance(xy1,xy2):
    (x1,y1),(x2,y2) = xy1,xy2
    if manhattan:
        return abs(x1-x2)+abs(y1-y2)
    else: # return euclidean distance
        return math.sqrt(math.pow(x1-x2,2)+math.pow(y1-y2,2))

# returns the index of the closest point
def closest(xys,xy1):
    return min(range(len(xys)), key=lambda i: distance(xy1,xys[i]))

def center(xys): # returns the average x and y
    return tuple([sum(l)/len(l) for l in zip(*xys)])

clusters=defaultdict(list)

#################################
# Main loop
#################################

# loop ten times at most
for _ in range(100):
    # calculate clusters
    clusters = defaultdict(list)
    for point in data:
        nearest_center = closest(centers, point)
        clusters[nearest_center].append(point)

    print(f"Centers: {centers}")
    print(f"Resulting Clusters: {dict(clusters)}")
    print("Calculating new centers...\n")

    # calculate new centers
    new_centers = [center(cluster) for cluster in clusters.values()]

    # break, if the centers have not changed
    if sorted(new_centers) == sorted(centers):
        break

    # set new centers for next iteration
    centers = new_centers

print(f"Final centers: {centers}")

#################################
# Plot of final state
#################################
x,y,col=[],[],[]
for c in clusters:
    for xy in clusters[c]:
        x.append(xy[0])
        y.append(xy[1])
        col.append(c)
plt.scatter(x, y, c=col, marker="x")
cx,cy = zip(*centers)
plt.scatter(cx,cy, marker="+", c="red")
plt.show()