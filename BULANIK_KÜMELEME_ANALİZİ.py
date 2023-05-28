from sklearn.cluster import KMeans
import numpy as np

# Veri k�mesini olu�turma
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# Bulan�k k�meleme modelini olu�turma
k = 2  # K�me say�s�
fcm = KMeans(n_clusters=k)

# Modeli veri k�mesine uygulama
fcm.fit(X)

# Sonu�lar� al�p k�meleri g�r�nt�leme
centers = fcm.cluster_centers_
labels = fcm.labels_

print("K�me merkezleri:")
print(centers)
print("K�me etiketleri:")
print(labels)
