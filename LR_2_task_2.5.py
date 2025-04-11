from pandas import read_csv
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import RidgeClassifier
import seaborn as sns


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width',
'class']
dataset = read_csv(url, names=names)

# Розділення датасету на навчальну та контрольну вибірки
array = dataset.values

# Вибір перших 4-х стовпців
X = array[:,0:4]

# Вибір 5-го стовпця
y = array[:,4]

#  Розділення даних на навчальні і контрольні
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Створюємо модель RidgeClassifier
model = RidgeClassifier()

# Оцінювання з cross-validation
kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
print("RidgeClassifier: %f (%f)" % (cv_results.mean(), cv_results.std()))  

model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Оцінюємо прогноз
print("Accuracy:", np.round(accuracy_score(Y_validation, predictions),4))
print("Матриця помилок:\n", np.round(confusion_matrix(Y_validation, predictions),4))
print('Коефіцієнт Каппа Коена:', np.round(cohen_kappa_score(Y_validation,predictions),4))
print('Коефіцієнт кореляції Метьюза:', np.round(matthews_corrcoef(Y_validation,predictions),4))
print("Звіт про класифікацію:\n", classification_report(Y_validation, predictions))

# Тестові дані
X_new = np.array([[5, 2.9, 1, 0.2]])

prediction = model.predict(X_new)
print("Прогноз моделі: {}".format(prediction[0]))

sns.set_theme()
mat = confusion_matrix(Y_validation, predictions)
sns.heatmap(mat.T, square = True, annot = True, fmt = 'd', cbar = False)
pyplot.xlabel('true label')
pyplot.ylabel('predicted label')
pyplot.title('Confusion Matrix')
pyplot.savefig("Confusion.jpg")