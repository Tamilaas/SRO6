# SRO6
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
# Бинарная классификация: набор данных Breast Cancer
X_bc, y_bc = load_breast_cancer(return_X_y=True)
X_bc_train, X_bc_test, y_bc_train, y_bc_test = train_test_split(X_bc, y_bc, test_size=0.2, random_state=42)
# Инициализация и обучение модели для бинарной классификации
model_bc = RandomForestClassifier(n_estimators=100, random_state=42)
model_bc.fit(X_bc_train, y_bc_train)
y_bc_pred = model_bc.predict(X_bc_test)
# Оценка производительности модели для бинарной классификации
accuracy_bc = accuracy_score(y_bc_test, y_bc_pred)
report_bc = classification_report(y_bc_test, y_bc_pred)
print("Binary Classification - Accuracy:", accuracy_bc)
print("Binary Classification - Classification Report:\n", report_bc)

# Многоклассовая классификация: набор данных Iris
X_multiclass, y_multiclass = load_iris(return_X_y=True)
X_multiclass_train, X_multiclass_test, y_multiclass_train, y_multiclass_test = train_test_split(X_multiclass, y_multiclass, test_size=0.2, random_state=42)
# Инициализация и обучение модели для многоклассовой классификации
model_multiclass = SVC(kernel='linear', C=1.0)
model_multiclass.fit(X_multiclass_train, y_multiclass_train)
y_multiclass_pred = model_multiclass.predict(X_multiclass_test)
# Оценка производительности модели для многоклассовой классификации
accuracy_multiclass = accuracy_score(y_multiclass_test, y_multiclass_pred)
report_multiclass = classification_report(y_multiclass_test, y_multiclass_pred)
print("Multiclass Classification - Accuracy:", accuracy_multiclass)
print("Multiclass Classification - Classification Report:\n", report_multiclass)
