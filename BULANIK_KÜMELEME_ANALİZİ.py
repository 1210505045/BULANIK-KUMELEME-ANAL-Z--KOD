from sklearn.cluster import KMeans
import numpy as np

# Veri kümesini oluþturma
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# Bulanýk kümeleme modelini oluþturma
k = 2  # Küme sayýsý
fcm = KMeans(n_clusters=k)

# Modeli veri kümesine uygulama
fcm.fit(X)

# Sonuçlarý alýp kümeleri görüntüleme
centers = fcm.cluster_centers_
labels = fcm.labels_

print("Küme merkezleri:")
print(centers)
print("Küme etiketleri:")
print(labels)
