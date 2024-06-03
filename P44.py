import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data_path = r'C:\Users\Pc\Desktop\Lab4AI\apartmentComplexData.txt'
data = pd.read_csv(data_path, header=None, delimiter=',')

data = data[[2, 3, 4, 5, 6, 8]]
data.columns = ['complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants', 'apartmentsNr', 'medianComplexValue']

X = data.drop('medianComplexValue', axis=1)
y = data['medianComplexValue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ridge_model = Ridge(alpha=1)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

lasso_model = Lasso(alpha=1)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)

elastic_net_model = ElasticNet(alpha=1, l1_ratio=0.5)
elastic_net_model.fit(X_train, y_train)
y_pred_elastic_net = elastic_net_model.predict(X_test)

ridge_mse = mean_squared_error(y_test, y_pred_ridge)
ridge_mae = mean_absolute_error(y_test, y_pred_ridge)
ridge_r2 = r2_score(y_test, y_pred_ridge)

lasso_mse = mean_squared_error(y_test, y_pred_lasso)
lasso_mae = mean_absolute_error(y_test, y_pred_lasso)
lasso_r2 = r2_score(y_test, y_pred_lasso)

elastic_net_mse = mean_squared_error(y_test, y_pred_elastic_net)
elastic_net_mae = mean_absolute_error(y_test, y_pred_elastic_net)
elastic_net_r2 = r2_score(y_test, y_pred_elastic_net)

print("Ridge Model:")
print("Mean Squared Error:", ridge_mse)
print("Mean Absolute Error:", ridge_mae)
print("R-squared:", ridge_r2)
print()

print("Lasso Model:")
print("Mean Squared Error:", lasso_mse)
print("Mean Absolute Error:", lasso_mae)
print("R-squared:", lasso_r2)
print()

print("Elastic Net Model:")
print("Mean Squared Error:", elastic_net_mse)
print("Mean Absolute Error:", elastic_net_mae)
print("R-squared:", elastic_net_r2)

fig, axes = plt.subplots(3, 1, figsize=(8, 12))

axes[0].scatter(y_test, y_pred_ridge, label='Predictions', color='blue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red', label='Actual')
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')
axes[0].set_title('Ridge Model')
axes[0].legend()

axes[1].scatter(y_test, y_pred_lasso, label='Predictions', color='green')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red', label='Actual')
axes[1].set_xlabel('Actual')
axes[1].set_ylabel('Predicted')
axes[1].set_title('Lasso Model')
axes[1].legend()

axes[2].scatter(y_test, y_pred_elastic_net, label='Predictions', color='orange')
axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red', label='Actual')
axes[2].set_xlabel('Actual')
axes[2].set_ylabel('Predicted')
axes[2].set_title('Elastic Net Model')
axes[2].legend()

plt.tight_layout()
plt.show()
