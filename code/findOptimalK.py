import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt

file = np.genfromtxt('/path/to/gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)

scaler = StandardScaler()
angle = scaler.fit_transform(angle)

train_angle, test_angle, train_label, test_label = train_test_split(angle, label, test_size=0.2, random_state=42)

# 다양한 k 값에 대한 성능 평가
k_values = range(1, 21)
cv_scores = []

for k in k_values:
    knn = cv2.ml.KNearest_create()
    knn.setDefaultK(k)
    scores = []
    for train_idx, val_idx in KFold(n_splits=5).split(train_angle):
        knn.train(train_angle[train_idx], cv2.ml.ROW_SAMPLE, train_label[train_idx])
        ret, results, neighbours, dist = knn.findNearest(train_angle[val_idx], k)
        predicted_label = results.ravel()
        accuracy = accuracy_score(train_label[val_idx], predicted_label)
        scores.append(accuracy)
    cv_scores.append(np.mean(scores))

# 최적의 k 값 찾기
optimal_k = k_values[np.argmax(cv_scores)]
print(f"Optimal k: {optimal_k}")

plt.plot(k_values, cv_scores)
plt.xlabel('k')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Optimal k value for KNN')
plt.show()
