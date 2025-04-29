import numpy as np
from itertools import cycle
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

# Шлях до файлу із даними
X = np.loadtxt('/content/drive/My Drive/Colab Notebooks/data_clustering.txt', delimiter=",")

# Оцінка ширини вікна для Х
bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

# Кластеризація даних методом зсуву середнього
meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

# Витягуємо центри всіх кластерів
cluster_centers = meanshift_model.cluster_centers_
print('\nCenters of clusters:\n', cluster_centers)

# Оцінка кількості кластерів
labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))
print("\nNumber of clusters in input data =", num_clusters)

# Візуалізуємо точки даних та центри кластерів
plt.figure()
markers = 'o*xvs'

# Задамо кольори кластерів
marker_color = ['red', 'black', 'yellow', 'green', 'blue']
for i, marker in zip(range(num_clusters), markers):
    # Відображення точок, що належать до поточного кластера
    plt.scatter(X[labels == i, 0], X[labels == i, 1], marker=marker, color=marker_color[i])

    # Відображення центру поточного кластера
    cluster_center = cluster_centers[i]
    plt.plot(cluster_center[0], cluster_center[1], marker='o',
             markerfacecolor='black', markeredgecolor='black', markersize=15)

plt.title('Кластери')
plt.show()