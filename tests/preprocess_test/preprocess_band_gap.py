import nanodesclib.classes
from nanodesclib.Calculation_to_dataframe import *
from nanodesclib.preprocess import *
import pubchempy as pcp
from pymatgen.core import Composition


my_df = pd.read_csv(r'C:\Users\Public\Documents\nanomaterials_descriptors_library\tests\dataset3_1.csv', delimiter=',')
df_0 = pd.read_csv(r'C:\Users\Public\Documents\nanomaterials_descriptors_library\tests\dataset3.csv', delimiter=',')
for i in df_0.index:
    df_0.loc[i, 'Metal oxide'] = df_0.loc[i, 'Metal oxide'].strip('-abr')
my_df['Eg (eV) (Exp.)'] = None
for i in my_df.index:

    my_df.loc[i, 'Eg (eV) (Exp.)'] = df_0[df_0['Metal oxide'] == my_df.loc[i, 'Metal oxide']].loc[df_0[df_0['Metal oxide'] == my_df.loc[i, 'Metal oxide']].index[0], 'Eg (eV) (Exp.)']


builder = DescriptorDatasetBuilder(dataframe=my_df, formula_col="Metal oxide")
result = builder.build()
result.to_csv('df_ds3.csv', index=False)

preprocessor = DataPreprocessor(
    target_column="Eg (eV) (Exp.)",
    drop_nan_threshold=0.3,
    use_knn_imputer=True,
    encoding="ordinal",
    scaling="standard"
)
clean_df = preprocessor.fit_transform(result)

clean_df.to_csv('clean_df_ds3.csv', index=False)