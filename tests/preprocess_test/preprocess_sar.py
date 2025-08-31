import nanodesclib.classes
from nanodesclib.Calculation_to_dataframe import *
from nanodesclib.preprocess import *
import pubchempy as pcp
import pymatgen as pmg

my_df = pd.read_csv(r'C:\Users\Public\Documents\nanomaterials_descriptors_library\tests\EDA\SAR_raw_data.csv')

my_df['smiles_1'] = None
my_df['smiles_2'] = None
my_df['smiles_3'] = None

for i in my_df['np_shell_1'].index:
    try:
        [pmg.Composition(j) for j in nanodesclib.classes.assign_class(my_df.loc[i, 'np_shell_1']).consist()]
        my_df.loc[i, 'np_core'] = '@'.join([my_df.loc[i, 'np_core'], my_df.loc[i, 'np_shell_1']])
    except:
        try:
            my_df.loc[i, 'smiles_1'] = pcp.get_compounds(my_df.loc[i, 'np_shell_1'], 'name')[0].isomeric_smiles
        except:
            continue

for j in my_df['np_shell_2'].index:
    try:
        [pmg.Composition(j) for i in nanodesclib.classes.assign_class(my_df.loc[j, 'np_shell_2']).consist()]
        my_df.loc[j, 'np_core'] = '@'.join([my_df.loc[j, 'np_core'], my_df.loc[j, 'np_shell_2']])
    except:
        try:
            my_df.loc[j, 'smiles_2'] = pcp.get_compounds(my_df.loc[j, 'np_shell_2'], 'name')[0].isomeric_smiles
        except:
            continue

for j in my_df['np_shell_3'].index:
    try:
        [pmg.Composition(i) for i in nanodesclib.classes.assign_class(my_df.loc[j, 'np_shell_3']).consist()]
        my_df.loc[j, 'np_core'] = '@'.join([my_df.loc[j, 'np_core'], my_df.loc[j, 'np_shell_3']])
    except:
        try:
            my_df.loc[j, 'smiles_3'] = pcp.get_compounds(my_df.loc[j, 'np_shell_3'], 'name')[0].isomeric_smiles
        except:
            continue

builder = DescriptorDatasetBuilder(dataframe=my_df, formula_col="np_core", smiles_cols=['smiles_1', 'smiles_2', 'smiles_3'])
result = builder.build()
result.to_csv('df_sar.csv', index=False)

preprocessor = DataPreprocessor(
    target_column="htherm_sar",
    drop_nan_threshold=0.2,
    columns_to_drop=['paper_doi', 'paper_files', 'paper_comment', 'syn_description'],
    use_knn_imputer=True,
    encoding="target",
    scaling="minmax"
)
clean_df = preprocessor.fit_transform(result)

clean_df.to_csv('clean_df_sar.csv', index=False)