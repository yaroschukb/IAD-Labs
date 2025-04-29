import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Завантажуємо набір даних Iris
iris = load_iris()
X = iris.data  
y = iris.target  

# Створюємо модель KMeans
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)

# Навчання моделі
kmeans.fit(X)

# Отримання передбачених міток кластерів
y_kmeans = kmeans.predict(X)

# Візуалізація результатів кластеризації (по перших двох ознаках)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=50)
centers = kmeans.cluster_centers_
flower_names = ['Setosa', 'Versicolor', 'Virginica']
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
# Додаємо підписи до центрів кластерів
for idx in range(len(centers)):
    plt.text(centers[idx, 0], centers[idx, 1], flower_names[idx],
             fontsize=12, ha='center', va='center', color='white',
             bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.5'))
plt.title('Кластеризація K-середніх для набору Iris')
plt.xlabel('Довжина чашолистка')
plt.ylabel('Ширина чашолистка')
plt.grid(True)
plt.show()