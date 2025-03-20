import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize

input_data = np.array([[-5.3, -8.9, 4.2],
                        [2.9, -5.0, -3.3],
                        [3.1, -2.8, -3.2],
                        [4.2, -1.4, 6.1]])
# 1. Бінаризація, номер варіанту 29
data_binarized = Binarizer(threshold=1.5).transform(input_data)
print("\n1. Binarized data:\n", data_binarized)

# 2. Стандартизовуємо набір даних
data_scaled = scale(input_data)
print("\n2. Standardized dataset and mean exclusion:")
# Виведення на екран середнього значення та стандартного відхилення
print("\nBEFORE: ")
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))

# Виключення среднього
print("\nAFTER: ")
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))

# 3. Масштабування
data_scaler_minmax = MinMaxScaler()
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\n3. Scaled data:")
print("\nМin max scaled data:\n", data_scaled_minmax)


# 4. Нормалізація даних
data_normalized_l1 = normalize(input_data, norm='l1')
data_normalized_l2 = normalize(input_data, norm='l2')
print("\n4. Normalized data:")
print("\nL1 normalized data:\n", data_normalized_l1)
print("\nL2 normalized data:\n", data_normalized_l2)