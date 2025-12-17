# train_model.py
import pandas as pd
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# 1) load dataset  loan_dataset1 = pd.read_csv("C:/Users/shiwa/Downloads/archive/train_u6lujuX_CVtuZ9i.csv")
# In train_model.py, line 10 should become:
loan_dataset1 = pd.read_csv("C:/Users/shiwa/Downloads/synthetic_loan_dataset_1M.csv")


# loan_dataset2 = pd.read_csv("C:/Users/shiwa/Downloads/train.csv")
# loan_dataset3 = pd.read_csv("C:/Users/shiwa/Downloads/loan_dataset_1000.csv")
df = pd.concat([loan_dataset1], ignore_index=True)
# drop rows with NA (same as your notebook)
df = df.dropna()

# 2) encode as in your notebook
df.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)
# you used 4 for "3+" in the notebook — keep that mapping
df = df.replace({'Dependents': {'3+': 4}})
df.replace({
    'Married': {'No': 0, 'Yes': 1},
    'Gender': {'Male': 1, 'Female': 0},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    'Education': {'Graduate': 1, 'Not Graduate': 0}
}, inplace=True)

# 3) prepare X, y
X = df.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']

# Save column order so backend uses same order
with open("columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

# 4) split and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2
)

clf = svm.SVC(kernel='linear', probability=False)  # same as your notebook
print("Model training ...")
clf.fit(X_train, y_train)
print("Model trained.")

# 5) test and save model
train_acc = accuracy_score(clf.predict(X_train), y_train)
test_acc = accuracy_score(clf.predict(X_test), y_test)
print("Train acc:", train_acc)
print("Test acc:", test_acc)

joblib.dump(clf, "loan_model.pkl")
print("Saved loan_model.pkl and columns.pkl")
