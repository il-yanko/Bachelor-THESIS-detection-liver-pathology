import seaborn as sns
import pandas as pd

df = pd.read_csv('./diplom_test/glcm_0.csv', sep='\t')

sns.pairplot(df, hue='pathology')
