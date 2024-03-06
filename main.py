import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

"""
За допомогою k-means побудувати модель кластеризації датасету
для цього
    встановити keras
    перетворити вхідні зображення в 1 вимірний вектор та нормалізувати семпли до [0, 1]
    побудувати модель кластеризації k-means для даних
    перевірити чи 10 - це оптимальна кількість кластерів для даного датасету (серед k=4, 8, 10, 12)
    візуалізувати кілька семплів із кожного кластеру (concat_images)
"""

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('Training Data: {}'.format(x_train.shape))
print('Training Labels: {}'.format(y_train.shape))
print("labels:", y_train)

# creating 1D array and normalization
x_train = x_train.reshape(x_train.shape[0], -1) / 255
x_test = x_test.reshape(x_test.shape[0], -1) / 255
print(x_train.shape, x_test.shape)

# creating model k-means
clusters = [4, 8, 10, 12]

sum_squared_errors = []
silhouette_scores = []
labels_opti_clust = None

for k in clusters:
    model = KMeans(n_clusters=k, random_state=0).fit(x_train)

    # elbow method
    sum_squared_errors.append(model.inertia_)

    # silhouette method
    labels = model.labels_
    silhouette_avg = silhouette_score(x_train, labels)
    silhouette_scores.append(silhouette_avg)

    # optimal clusters
    if k == 8:
        labels_opti_clust = labels

# creating plot for elbow method
plt.plot(clusters, sum_squared_errors, marker='o')
plt.xlabel('number of clusters')
plt.ylabel('sum of squared errors')
plt.title('Elbow method')
plt.savefig('elbow_method.png')
plt.show()

# creating plot for silhouette method
plt.plot(clusters, silhouette_scores, marker="o")
plt.xlabel('number of clusters')
plt.ylabel('silhouette coefficient')
plt.title('Silhouette method')
plt.savefig('silhouette_method.png')
plt.show()


# concatenate images horizontally
def concat_images(imgs_list):
    return np.concatenate(imgs_list, axis=1)


# sample of clusters
sample = []
for i in range(8):
    # take 4 samples for each clusters
    cluster_samples = x_train[labels_opti_clust == i][:4]
    # change shape img to 28x28
    reshaped_samples = [np.reshape(img, (28, 28)) for img in cluster_samples]
    sample.extend(reshaped_samples)

combined_image = concat_images(sample)
plt.imshow(combined_image)
plt.show()
