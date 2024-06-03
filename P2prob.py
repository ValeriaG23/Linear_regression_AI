import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

data_path = r'C:\Users\Pc\Desktop\Lab4AI\apartmentComplexData.txt'
df = pd.read_csv(data_path, header=None)

column_names = ['Vechimea_complexului', 'NrTotalDeCamere', 'NrDeDormitoare', 'NrDeLocuitoriAlComplexului', 'NrDeApartamente', 'ValoareaMedianaAlComplexului', 'Elevator', 'Parcare', 'Suprafata']

num_columns = df.shape[1]
if num_columns == 9:
    df.columns = column_names
else:
    print("Numărul de coloane în DataFrame nu este cel așteptat.")

# Definim variabila independentă (X) și variabila dependentă (y)
X = df.drop('ValoareaMedianaAlComplexului', axis=1)
y = df['ValoareaMedianaAlComplexului']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

# Antrenăm modelul folosind setul de antrenare
model.fit(X_train, y_train)

# Facem predicții folosind setul de testare
y_pred = model.predict(X_test)

# Evaluăm performanța modelului
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)

plt.scatter(y_test, y_pred)
plt.xlabel("Valoarea reală")
plt.ylabel("Valoarea prezisă")
plt.title("Rezultatele modelului de regresie liniară")
plt.show()
