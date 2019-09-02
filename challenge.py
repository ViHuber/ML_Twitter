from sklearn import datasets
import pandas as pd

boston = datasets.load_boston()
df = pd.DataFrame(boston.data)
df.columns = boston.feature_names
df['MEDV'] = boston.target
print(df.head())