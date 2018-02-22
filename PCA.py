#%% load modules
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

from adjustText import adjust_text

import matplotlib.pyplot as plt

import seaborn as sns

#%% import data
df_full = pd.read_table("./mosquitoes_spectra (170623).dat")
df_full["sp_feed"] = df_full["Species"] + " " + df_full["Status"]
# df_full.index= df_full["sp_feed"]

#%% PCA

pca_data = pd.concat([df_full["sp_feed"], df_full.iloc[:, 5:-1]], axis=1)
pca_data = pca_data.set_index("sp_feed")
# pca_data.rename(index={
#     "4mo uninfected": "4mo Uninf.",
#     "12mo uninfected": "12mo Uninf.",
#     "4mo infected": "4mo Inf.",
#     "12mo infected": "12mo Inf."}, inplace=True)

pca_data = pca_data.dropna(axis=1)

#scale data for ML
pca_data_scaled = pca_data.copy()
# center & scale to unit variance
pca_data_scaled[pca_data_scaled.columns] = StandardScaler().fit_transform( pca_data_scaled[pca_data_scaled.columns].as_matrix())

# df_age_pca = pca_data_scaled
n_classes = len(pca_data_scaled.index.unique())
seed = 4
pca = PCA(n_components=n_classes, random_state=seed).fit(pca_data_scaled)
reduced_data = PCA(n_components=2, random_state=seed).fit_transform(pca_data_scaled) 

# cluster
kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)

kmeans.fit(reduced_data)

#%% Plot the decision boundary. For that, we will assign a color to eachx_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1

# Step size of the mesh. Decrease to increase the quality of the VQ.
# point in the mesh [x_min, x_max]x[y_min, y_max].
h = (x_max - x_min) / 1000

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(7, 5))
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Dark2,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=10)

# labels = np.sort(list(pca_data_scaled.index))
# texts = []
# for label, x, y in zip(labels, reduced_data[:, 0], reduced_data[:, 1]):
#     texts.append(plt.text(x, y, label, fontsize="x-small",
#                           bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.4)))
# adjust_text(texts, force_text=0.7, force_points=0.5, arrowprops=dict(arrowstyle="->",
#                                                                      connectionstyle='arc3,rad=0.4'))

# for label, x, y in zip(labels, reduced_data[:, 0], reduced_data[:, 1]):
#     plt.annotate(
#         label,
#         xy=(x, y), xytext=(25, 15), fontsize="x-small",
#         textcoords='offset points', ha='right', va='bottom',
#         bbox=dict(boxstyle='round,pad=0.4', fc='yellow', alpha=0.4),
#         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.4'))

# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on mosquito MIR spectra\n(PCA-reduced data)')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())


plt.savefig("./plots/pca_kmeans.pdf", bbox_inches="tight")
