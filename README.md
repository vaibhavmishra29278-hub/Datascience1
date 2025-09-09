1.1 data imputation
import pandas as pd [cite: 7]
import matplotlib.pyplot as plt [cite: 7]
df = pd.read_csv("Namrata/titanic_toy.csv") [cite: 7, 9]
df [cite: 7]
df.dropna() [cite: 9]
age_mean = df["Age"].mean() [cite: 13]
age_median = df["Age"].median() [cite: 13]
age_mode = df["Age"].mode()[0] [cite: 13]

df["Age_mean"] = df['Age'].fillna(age_mean) [cite: 14]
df["Age_median"] = df['Age'].fillna(age_median) [cite: 14]
df["Age_mode"] = df['Age'].fillna(age_mode) [cite: 14]
df [cite: 14]
df['Age'].plot(kind='kde', label="Age Before") [cite: 15]
df["Age_mean"].plot(kind='kde') [cite: 15]
df["Age_median").plot(kind='kde') [cite: 15]
df["Age_mode"].plot(kind='kde') [cite: 15]
plt.legend() [cite: 15]
1.2 handeling categorical 
import pandas as pd [cite: 15]
from sklearn.preprocessing import LabelEncoder [cite: 15]
df = pd.read_csv("Namrata/Social_Network_Ads.csv") [cite: 15]
df [cite: 15]
df.drop("User ID", axis=1, inplace=True) [cite: 18]
df [cite: 18]
pd.get_dummies(df, columns=["Gender"], dtype=int) [cite: 19]
le = LabelEncoder() [cite: 21]
df["Gender"] = le.fit_transform(df["Gender"]) [cite: 21]
df [cite: 21]
1.3 Feature Scalling
import pandas as pd [cite: 23]
from sklearn.preprocessing import StandardScaler, MinMaxScaler [cite: 23]

df = pd.read_csv("Namrata/titanic_toy.csv") [cite: 23]
print("Original data:\n", df.head()) [cite: 23]

df["Age"].fillna(df["Age"].median(), inplace=True) [cite: 23]
df["Fare"].fillna(df["Fare"].median(), inplace=True) [cite: 23]
print("\nAfter filling Missing values:\n", df.head()) [cite: 23]

scaler_standard = StandardScaler() [cite: 23]
df_standard = df.copy() [cite: 23]
df_standard[["Age", "Fare", "Family"]] = scaler_standard.fit_transform(df[["Age", "Fare", "Family"]]) [cite: 23]
print("\nStandardized Data:\n", df_standard.head()) [cite: 23]

scaler_minmax = MinMaxScaler() [cite: 23]
df_minmax = df.copy() [cite: 23]
df_minmax[["Age", "Fare", "Family"]] = scaler_minmax.fit_transform(df[["Age", "Fare", "Family"]]) [cite: 23]
print("\nNormalized Data:\n", df_minmax.head()) [cite: 23]
1.4 Feature Selection
from sklearn.datasets import load_diabetes [cite: 23]
import pandas as pd [cite: 23]
import seaborn as sb [cite: 23]
from sklearn.feature_selection import VarianceThreshold [cite: 23]

data = load_diabetes() [cite: 23]
df = pd.DataFrame(data.data, columns=data.feature_names) [cite: 23]
df['target'] = data.target [cite: 23]

print(df.corr()) [cite: 23]
dataplot = sb.heatmap(df.corr(), annot=True) [cite: 23]

selector = VarianceThreshold() [cite: 23]
selector.fit_transform(df) [cite: 23]

selector = VarianceThreshold(threshold=0.002) [cite: 23]
selector.fit_transform(df) [cite: 23]
2.1 Simple linear
import pandas as pd [cite: 23]
import matplotlib.pyplot as plt [cite: 23]
from sklearn.preprocessing import StandardScaler, MinMaxScaler [cite: 23]
import seaborn as sns [cite: 23]
import numpy as np [cite: 23]
from sklearn.linear_model import LinearRegression [cite: 23]
from sklearn.model_selection import train_test_split [cite: 23]

df = pd.read_csv("Namrata/Ice_Cream.csv") [cite: 23]
df.head() [cite: 23]
sns.boxplot(df['Temperature']) [cite: 23]
sns.boxplot(df['Revenue']) [cite: 23]

X = df["Temperature"].values.reshape(-1, 1) [cite: 23]
Y = df["Revenue"].values.reshape(-1, 1) [cite: 23]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 12) [cite: 23]

lr = LinearRegression() [cite: 23]
lr.fit(X_train, y_train) [cite: 23]

m = lr.coef_ [cite: 23]
c = lr.intercept_ [cite: 23]

y_pred = m * X + c [cite: 23]

plt.scatter(df["Temperature"], df["Revenue"]) [cite: 23]
plt.plot(df["Temperature"], y_pred, color='red') [cite: 23]
plt.show() [cite: 23]
2.2 multiple
from sklearn.linear_model import LinearRegression [cite: 24]
import pandas as pd [cite: 24]
import plotly.express as px [cite: 24]
from sklearn.datasets import make_regression [cite: 24]

X, y = make_regression(n_samples=100, n_features=2, random_state=2, n_targets=1) [cite: 24]
df = pd.DataFrame(X, columns=["x1", "x2"]) [cite: 24]
df['y'] = y [cite: 24]
df [cite: 24]

lr = LinearRegression() [cite: 24]
lr.fit(X, y) [cite: 24]

print("Beta1 and Beta2 :", lr.coef_) [cite: 24]
print("Beta_0 ::", lr.intercept_) [cite: 24]

px.scatter_3d(df, x="x1", y="x2", z="y") [cite: 24]
2.3 Polynomial
from sklearn.preprocessing import PolynomialFeatures [cite: 24]
from sklearn.linear_model import LinearRegression [cite: 24]
from sklearn.metrics import mean_squared_error, r2_score [cite: 24]
from sklearn.preprocessing import StandardScaler [cite: 24]
from sklearn.pipeline import Pipeline [cite: 24]
import pandas as pd [cite: 24]
import numpy as np [cite: 24]
import matplotlib.pyplot as plt [cite: 24]

df = pd.read_csv("Namrata/Poly_Data.csv") [cite: 24]
df.drop("Unnamed: 0", axis=1, inplace=True) [cite: 24]
2.4 Gradient descent
import numpy as np [cite: 24]
import pandas as pd [cite: 24]
import matplotlib.pyplot as plt [cite: 24]
from sklearn.datasets import make_regression [cite: 24]

X, y = make_regression(n_samples=100, n_features=1, noise=20) [cite: 24]
plt.scatter(X, y) [cite: 24]
plt.show() [cite: 24]

from sklearn.linear_model import LinearRegression [cite: 24]
from sklearn.metrics import mean_squared_error, r2_score [cite: 24]

lin_reg = LinearRegression() [cite: 24]
lin_reg.fit(X, y) [cite: 24]
print(lin_reg.coef_) [cite: 24]
print(lin_reg.intercept_) [cite: 24]

y_pred = lin_reg.predict(X) [cite: 24]
plt.plot(X, y_pred, color='red') [cite: 24]
plt.scatter(X, y) [cite: 24]
plt.show() [cite: 24]

# Manual Gradient Descent
x1 = X [cite: 26]
y1 = y [cite: 26]
m = 0 [cite: 26]
c = 0 [cite: 26]
L = 0.001 [cite: 26]
epochs = 100 [cite: 26]

for i in range(epochs): [cite: 26]
    y_pred = m * x1 + c [cite: 26]
    D_m = (-2 / len(x1)) * sum(x1 * (y1 - y_pred)) [cite: 26]
    D_c = (-2 / len(x1)) * sum(y1 - y_pred) [cite: 26]
    m = m - L * D_m [cite: 26]
    c = c - L * D_c [cite: 26]

y_pred = m * x1 + c [cite: 26]
print("Slope:", m, "Intercept:", c) [cite: 26]
plt.scatter(x1, y1) [cite: 26]
plt.plot(x1, y_pred, color='black') [cite: 26]
plt.show() [cite: 26]
3.1 L1 lasso
from sklearn.datasets import make_regression
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso

X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
df = pd.DataFrame(X, columns=["x1"])
df["y"] = y
df

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)

print("Lasso Coeficient:", lasso_reg.coef_)
print("Lasso Intercept:", lasso_reg.intercept_)
3.2 l2 lasso
from sklearn.datasets import make_regression
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
df = pd.DataFrame(X, columns=["x1"])
df["y"] = y
df

ridge_reg = Ridge(alpha=0.1)
ridge_reg.fit(X, y)

print("Ridge Coeficient:", ridge_reg.coef_)
print("Ridge Intercept:", ridge_reg.intercept_)
4.1 Logistic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("Namrata/Social_Network_Ads.csv")
df.head()

X = df.drop(["User ID", "Purchased"], axis=1)
y = df["Purchased"]

X = pd.get_dummies(X, columns=["Gender"], dtype=int)
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")
4.2 Decision TRee
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("Namrata/iris.csv")
df.head()
df.info()

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
4.3 Random Forest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("Namrata/iris.csv")

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
4.4 Naive Bayes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

df = pd.read_csv("Namrata/iris.csv")

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
4.5 Support vector machine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("Namrata/iris.csv")

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
