import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn import tree
from IPython.display import Image

# ======= ЧАСТИНА 1 =======

# Створення фрейму даних
response = requests.get('https://raw.githubusercontent.com/yzghurovskyi/LoanPredictionProblem/refs/heads/main/titanic-train.csv')
passengers = pd.read_csv(StringIO(response.text))
pd.set_option('display.max_rows', 10)
passengers.head(10)

# Підготовка даних для побудови моделі дерева рішень.

passengers.info()

# Заміна статі на числову мітку
passengers["Gender"] = passengers["Gender"].apply(lambda x: 0 if x == "male" else 1)

# Перевіримо, чи Gender замінено
passengers[["Name", "Gender"]].head(25)

# Заміна статі на числову мітку
passengers["Gender"] = passengers["Gender"].apply(lambda x: 0 if x == "male" else 1)

# Перевіримо, чи Gender замінено
passengers[["Name", "Gender"]].head(25)

# Заповнення пропущених значень у колонці age середнім значенням
passengers["Age"].fillna(passengers["Age"].mean(), inplace=True)

# Навчання моделі
y_target = passengers["Survived"].values
columns = ["Fare", "Pclass", "Gender", "Age", "SibSp"]
X_input = passengers[columns].values

# Створюємо модель класифікатора дерева рішень
clf_train = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf_train = clf_train.fit(X_input, y_target)

# Оцінка моделі
clf_train.score(X_input, y_target)

# Візуалізація дерева

from sklearn.tree import export_graphviz

with open("./titanic.dot", 'w') as f:
    export_graphviz(clf_train, out_file=f, feature_names=columns)
    
Image("./titanic.png")

# ======= ЧАСТИНА 2 =======
# Завантажуємо тестові дані
response = requests.get('https://raw.githubusercontent.com/yzghurovskyi/LoanPredictionProblem/refs/heads/main/titanic-test.csv')
trainData = pd.read_csv(StringIO(response.text))
pd.set_option('display.max_rows', 10)

trainData.info()

# Замінити значення "male" та "female" на 0 для чоловіків і 1 для жінок.
trainData["Gender"] = trainData["Gender"].apply(lambda x: 0 if x == "male" else 1)

# Перевіримо, чи Gender замінено
trainData[["Name", "Gender"]].head(25)

# Заповнення пропущених значень у колонці age середнім значенням
trainData["Age"].fillna(trainData["Age"].mean(), inplace=True)

# Перевіримо чи заповнились дані
trainData.info()

X_input = trainData[columns].values

# Створюємо модель класифікатора дерева рішень

target_labels = clf_train.predict(X_input)

# Створення датафрейму з передбаченнями та іменами пасажирів
target_labels = pd.DataFrame({
    'Est_Survival': target_labels,
    'Name': trainData['Name']
})

# Перегляд результату
target_labels.head()

# Завантажуємо основну істину щодо виживання кожного пасажира
all_data = pd.read_csv("https://raw.githubusercontent.com/yzghurovskyi/LoanPredictionProblem/refs/heads/main/titanic_all.csv")

# об’єднуємо датафрейм з передбаченнями з реальними мітками
testing_results = pd.merge(target_labels, all_data[['Name','Survived']], on=['Name'])

# Обчислюємо точність як частку правильних передбачень
acc = np.sum(testing_results['Est_Survival'] == testing_results['Survived']) / float(len(testing_results))
print("Точність моделі", acc)

# Завантажуємо лише потрібні стовпці з titanic_all.csv
all_data = pd.read_csv(
    "https://raw.githubusercontent.com/yzghurovskyi/LoanPredictionProblem/refs/heads/main/titanic_all.csv",
    usecols=['Survived', 'Pclass', 'Gender', 'Age', 'SibSp', 'Fare']
)

# Перевіримо кількість записів та пропуски
print("Кількість записів:", all_data.shape[0])
print("Пропущені значення:\n", all_data.isnull().sum())

all_data["Gender"] = all_data["Gender"].apply(lambda x: 0 if x == 'male' else 1)

# Заповнення пропущених значень у віці середнім значенням
all_data["Age"] = all_data["Age"].fillna(all_data["Age"].mean())

# Перевіримо результат
all_data.head()

# Створення навчальних та тестових даних
from sklearn.model_selection import train_test_split
from sklearn import tree

# Вхідні ознаки та цільова змінна
columns = ["Fare", "Pclass", "Gender", "Age", "SibSp"]
X = all_data[columns].values
y = all_data["Survived"].values

# Розділення даних на 60% для навчання, 40% для тесту
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=0)

# Створюємо класифікатор дерева рішень
clf_train = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Навчаємо модель на тренувальних даних
clf_train = clf_train.fit(X_train, y_train)

# Оцінка точності моделі на тренувальному та тестовому наборах
train_score = clf_train.score(X_train, y_train)
test_score = clf_train.score(X_test, y_test)

# Вивід результату
print("Training score =", round(train_score, 4), "| Testing score =", round(test_score, 4))