import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data_path = r'C:\Users\Pc\Desktop\Lab4AI\apartmentComplexData.txt'
data = pd.read_csv(data_path, header=None, delimiter=',')

data = data[[2, 3, 4, 5, 6, 8]]
data.columns = ['complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants', 'apartmentsNr', 'medianComplexValue']

# Definirea variabilelor independente (X) și a variabilei dependente (y)
X = data.drop('medianComplexValue', axis=1)
y = data['medianComplexValue']

# Împărțirea setului de date în set de antrenare și set de testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inițializarea modelului de regresie liniară
model = LinearRegression()

# Antrenarea modelului folosind setul de antrenare
model.fit(X_train, y_train)

# Previziuni pe setul de testare
y_pred = model.predict(X_test)

# Definirea caracteristicilor pentru casa nouă
new_house_features = {
    'complexAge': 50,
    'totalRooms': 3,
    'totalBedrooms': 1,
    'complexInhabitants': 1000,
    'apartmentsNr': 200,
    'medianComplexValue': 1,
}

# Transformăm caracteristicile casei noi într-un DataFrame
new_house_df = pd.DataFrame([new_house_features])

# Folosim modelul pentru a prezice valoarea mediană a complexului pentru casa nouă
predicted_price = model.predict(new_house_df.drop('medianComplexValue', axis=1))

print("Predicted price of the new house:", predicted_price[0])
