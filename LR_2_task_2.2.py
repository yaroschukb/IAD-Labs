import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Завантаження даних 
input_file = '/content/drive/My Drive/Colab Notebooks/census+income/adult.data'
X = []
count_class1 = 0
count_class2 = 0
max_datapoints = 2000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line.strip().split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)

# === Кодування ознак ===
label_encoders = []
X_encoded = np.empty(X.shape)
for i in range(X.shape[1]):
    try:
        X_encoded[:, i] = X[:, i].astype(float)
        label_encoders.append(None)
    except ValueError:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoders.append(le)

X_features = X_encoded[:, :-1].astype(float)
y_labels = X_encoded[:, -1].astype(int)

# === Розділення даних ===
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=5)
# Функція для тренування й оцінки з різними ядрами
def evaluate_svm(kernel_name, **kwargs):
    print(f"\n--- SVM з ядром: {kernel_name} ---")
    model = SVC(kernel=kernel_name, **kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
    print("Precision:", round(precision_score(y_test, y_pred, average='weighted') * 100, 2), "%")
    print("Recall:", round(recall_score(y_test, y_pred, average='weighted') * 100, 2), "%")
    print("F1 Score:", round(f1_score(y_test, y_pred, average='weighted') * 100, 2), "%")

    # F1 score з крос-валідацією
    f1_cv = cross_val_score(model, X_features, y_labels, scoring='f1_weighted', cv=3)
    print("F1 score (CV):", str(round(100 * f1_cv.mean(), 2)) + "%")

    return model

# Тестові дані
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
              '0', '0', '40', 'United-States']

# Оцінювання для трьох ядер
models = {
    'poly': {'degree': 3},
    'rbf': {},
    'sigmoid': {}
}

for kernel_type, params in models.items():
    model = evaluate_svm(kernel_type, **params)

    # Кодування тестовиї даних
    input_data_encoded = []
    for i in range(len(input_data)):
        if input_data[i].isdigit():
            input_data_encoded.append(int(input_data[i]))
        else:
            le = label_encoders[i]
            input_data_encoded.append(int(le.transform([input_data[i]])[0]))

    input_data_encoded = np.array([input_data_encoded])

    # Прогнозування
    predicted_class = model.predict(input_data_encoded)
    label_le = label_encoders[-1]
    print("Прогнозована категорія для тестової точки:", label_le.inverse_transform(predicted_class)[0])
