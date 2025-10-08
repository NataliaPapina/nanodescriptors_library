import nanodesclib.classes
from nanodesclib.Calculation_to_dataframe import *
from nanodesclib.preprocess import *
import pubchempy as pcp
from pymatgen.core import Composition

my_df = pd.read_csv(r'C:\Users\Public\Documents\nanomaterials_descriptors_library\tests\EDA\SAR_raw_data.csv')

my_df['smiles_1'] = None
my_df['smiles_2'] = None
my_df['smiles_3'] = None

my_df[my_df['np_core'] == 'Co1Fe2O4@N-(trimethoxysilylpropyl)-ethylenediaminetriacetate: trimethoxysilylpropyl']['np_shell_1'] = '(trimethoxysilylpropyl)-ethylenediaminetriacetate'
my_df[my_df['np_core'] == 'Co1Fe2O4@N-(trimethoxysilylpropyl)-ethylenediaminetriacetate: trimethoxysilylpropyl']['np_core'] = 'Co1Fe2O4@N'

my_df[my_df['np_core'] == 'Mn1Fe2O4@N-(trimethoxysilylpropyl)-ethylenediaminetriacetate: trimethoxysilylpropyl']['np_shell_1'] = '(trimethoxysilylpropyl)-ethylenediaminetriacetate'
my_df[my_df['np_core'] == 'Mn1Fe2O4@N-(trimethoxysilylpropyl)-ethylenediaminetriacetate: trimethoxysilylpropyl']['np_core'] = 'Mn1Fe2O4@N'

my_df[my_df['np_core'] == 'Ni1Fe2O4@N-(trimethoxysilylpropyl)-ethylenediaminetriacetate: trimethoxysilylpropyl']['np_shell_1'] = '(trimethoxysilylpropyl)-ethylenediaminetriacetate'
my_df[my_df['np_core'] == 'Ni1Fe2O4@N-(trimethoxysilylpropyl)-ethylenediaminetriacetate: trimethoxysilylpropyl']['np_core'] = 'Ni1Fe2O4@N'

my_df.loc[my_df['np_core'] == 'Fe2.2C + Fe5C2', 'np_core'] = 'Fe2.2C/Fe5C2'
my_df.loc[my_df['np_core'] == 'yFe2O4', 'np_core'] = 'Fe2O4'

for i in my_df['np_shell_1'].index:
    try:
        [Composition(j) for j in nanodesclib.classes.assign_class(my_df.loc[i, 'np_shell_1']).consist()]
        my_df.loc[i, 'np_core'] = '@'.join([my_df.loc[i, 'np_core'], my_df.loc[i, 'np_shell_1']])
    except:
        try:
            my_df.loc[i, 'smiles_1'] = pcp.get_compounds(my_df.loc[i, 'np_shell_1'], 'name')[0].isomeric_smiles
        except:
            continue

for j in my_df['np_shell_2'].index:
    try:
        [Composition(j) for i in nanodesclib.classes.assign_class(my_df.loc[j, 'np_shell_2']).consist()]
        my_df.loc[j, 'np_core'] = '@'.join([my_df.loc[j, 'np_core'], my_df.loc[j, 'np_shell_2']])
    except:
        try:
            my_df.loc[j, 'smiles_2'] = pcp.get_compounds(my_df.loc[j, 'np_shell_2'], 'name')[0].isomeric_smiles
        except:
            continue

for j in my_df['np_shell_3'].index:
    try:
        [Composition(i) for i in nanodesclib.classes.assign_class(my_df.loc[j, 'np_shell_3']).consist()]
        my_df.loc[j, 'np_core'] = '@'.join([my_df.loc[j, 'np_core'], my_df.loc[j, 'np_shell_3']])
    except:
        try:
            my_df.loc[j, 'smiles_3'] = pcp.get_compounds(my_df.loc[j, 'np_shell_3'], 'name')[0].isomeric_smiles
        except:
            continue

builder = DescriptorDatasetBuilder(dataframe=my_df.loc[:,['np_core', 'smiles_1', 'smiles_2', 'smiles_3', 'htherm_sar']], formula_col="np_core", smiles_cols=['smiles_1', 'smiles_2', 'smiles_3'])
result = builder.build()
result.to_csv('df_sar_only_lb.csv', index=False)

preprocessor = DataPreprocessor(
    target_column="htherm_sar",
    drop_nan_threshold=0.3,
    use_knn_imputer=True,
    encoding="target",
    scaling='minmax'
)
clean_df = preprocessor.fit_transform(result)

clean_df.to_csv('clean_df_sar_only_lb.csv', index=False)