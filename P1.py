import pandas as pd
from matplotlib import pyplot as plt

data_path = r'C:\Users\Pc\Desktop\Lab4AI\apartmentComplexData.txt'

data = pd.read_csv(data_path, header=None, delimiter=',', usecols=range(6))
data.columns=['Vechimea_complexului', 'NrTotalDeCamere', 'NrDeDormitoare', 'NrDeLocuitoriAlComplexului', 'NrDeApartamente', 'ValoareaMedianaAlComplexului']


print(data.describe())

data.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()