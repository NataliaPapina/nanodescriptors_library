from nanodesclib.Calculation_to_dataframe import *
from nanodesclib.preprocess import *
import pubchempy as pcp

my_df = pd.read_csv(r'C:\Users\Public\Documents\nanomaterials_descriptors_library\tests\EDA\nanotox.csv')
my_df['smiles_1'] = None

for i in my_df['coat/functional group'].index:
    try:
        my_df.loc[i, 'smiles_1'] = pcp.get_compounds(my_df.loc[i, 'np_shell_1'], 'name')[0].isomeric_smiles
    except:
        continue

builder = DescriptorDatasetBuilder(dataframe=my_df.loc[:,["material", 'smiles_1', "viability (%)"]],
                                   formula_col="material", smiles_cols=['smiles_1'])
result = builder.build()
result.to_csv('df_nt_only_library.csv', index=False)

preprocessor = DataPreprocessor(
    target_column="viability (%)",
    drop_nan_threshold=0.2,
    use_knn_imputer=True,
    encoding="target",
    scaling="minmax"
)
clean_df = preprocessor.fit_transform(result)

clean_df.to_csv('clean_df_nt_only_library.csv', index=False)