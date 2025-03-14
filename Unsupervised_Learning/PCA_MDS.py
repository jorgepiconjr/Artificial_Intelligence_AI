### Data preprocessing
dataset = [
    ["Headache",  "Fever",  "Dry_cough",  "Tiredness",  "Conjunctivitis",   "Diarrhoea",  "Dehydration ",  "Rash",  "Cramps",  "Nausea", "Diagnosis"],
    ["Yes",  "Yes",   "Yes",   "Yes",  "Yes",  "Yes",    "No",    "Yes",  "No",   "No",   "Covid"],
    ["No",  "Yes",   "No",   "No",  "No",  "Yes",    "Yes",    "No",  "Yes",   "Yes",   "Food_poison"],
    ["Yes",  "No",   "Yes",   "Yes",  "Yes",  "No",    "No",    "Yes",  "No",   "No",   "Covid"],
    ["Yes",  "Yes",   "Yes",   "Yes",  "No",  "Yes",    "No",    "No",  "No",   "No",   "Covid"],
    ["Yes",  "Yes",   "No",   "No",  "No",  "Yes",    "Yes",    "No",  "Yes",   "Yes",   "Food_poison"],
    ["Yes",  "Yes",   "No",   "Yes",  "No",  "Yes",    "No",    "No",  "Yes",   "Yes",   "Food_poison"],
    ["Yes",  "Yes",   "Yes",   "No",  "No",  "Yes",    "No",    "Yes",  "Yes",   "No",   "Food_poison"],
    ["Yes",  "No",   "No",   "Yes",  "Yes",  "No",    "No",    "Yes",  "No",   "No",   "Covid"],
    ["Yes",  "Yes",   "No",   "Yes",  "Yes",  "No",    "Yes",    "Yes",  "Yes",   "Yes",   "Covid"],
    ["Yes",  "Yes",   "No",   "Yes",  "No",  "Yes",    "Yes",    "No",  "Yes",   "Yes",   "Food_poison"]
]
# Dictionary to assign numeric values
val2int =  {"Yes": 1, "No": 0}
# Define colors for the two classes
val2col = {"Food_poison": "blue", "Covid": "red"}

header = dataset[0] # header from data
data = dataset[1:]  # remove header from data

# Symptoms data as binaries
symptoms = [[val2int[val] for val in example[:-1]] for example in data]

# Class label (diagnosis) color for visualization
diagnosis = [val2col[example[-1]] for example in data]

#_________________________________________________________________________________________________
'''
Principal Component Analysis (PCA) mithilfe der Implementierung in scikit in 2 Dimensionen
'''
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

print("PCA mit zwei Dimensionen")
pca_2d = PCA(n_components=2)
data_2d = pca_2d.fit_transform(symptoms)

(x,y) = zip(*data_2d)
plt.scatter(x, y, c=diagnosis)
plt.show()
#_________________________________________________________________________________________________
for i in range(1,4):
    pca_id = PCA(n_components=i)
    data_id = pca_id.fit_transform(symptoms)
    print(f"{i}D: Explained Variance by for each dimension: {pca_id.explained_variance_ratio_}")

# PCA mit nur 1 Dimension
pca_1d = PCA(n_components=1)
data_1d = pca_1d.fit_transform(symptoms)

fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(121)
ax1.scatter(data_1d, [0]*len(data_1d),c=diagnosis)
ax1.set_title("PCA mit nur einer Dimension")

# PCA mit 3 Dimensionen
pca_3d = PCA(n_components=3)
data_3d = pca_3d.fit_transform(symptoms)

ax2 = fig.add_subplot(122, projection='3d')
(x,y,z) = zip(*data_3d)
ax2.scatter(x,y,z,c=diagnosis)
ax2.set_title("PCA mit drei Dimensionen")
plt.show()
#_________________________________________________________________________________________________
### Multidimensional Scalings (MDS)
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

print("MDS mit zwei Dimensionen")
mds = MDS(n_components=2)
mds_data = mds.fit_transform(symptoms)

# plot the results
fig = plt.figure()
ax = fig.add_subplot(111)
(x,y) = zip(*mds_data)
plt.scatter(x, y, c=diagnosis)

# label the points
for i in range(10):
    ax.annotate(i, (x[i]+0.03,y[i]))

# equal scale for x and y to better judge distances
ax.set_aspect('equal', adjustable='box')

plt.show()
print(f"Stress: {mds.stress_}")