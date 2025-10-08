import nanodesclib.classes
from nanodesclib.Calculation_to_dataframe import *
from nanodesclib.preprocess import *
import pubchempy as pcp
from pymatgen.core import Composition

my_df = pd.read_csv(r'C:\Users\Public\Documents\nanomaterials_descriptors_library\tests\EDA\mri_raw_data.csv')

my_df[my_df['np_core'] == 'Fe1Fe2O4@PEG-550']['np_shell_1'] = 'PEG-550'
my_df[my_df['np_core'] == 'Fe1Fe2O4@PEG-550']['np_core'] = 'Fe1Fe2O4'

my_df[my_df['np_core'] == 'Fe1Fe2O4@PEG-750']['np_shell_1'] = 'PEG-750'
my_df[my_df['np_core'] == 'Fe1Fe2O4@PEG-750']['np_core'] = 'Fe1Fe2O4'

my_df[my_df['np_core'] == 'Fe1Fe2O4@PEG-1000']['np_shell_1'] = 'PEG-1000'
my_df[my_df['np_core'] == 'Fe1Fe2O4@PEG-1000']['np_core'] = 'Fe1Fe2O4'

my_df[my_df['np_core'] == 'Fe1Fe2O4@PEG-2000']['np_shell_1'] = 'PEG-2000'
my_df[my_df['np_core'] == 'Fe1Fe2O4@PEG-2000']['np_core'] = 'Fe1Fe2O4'

my_df[my_df['np_core'] == 'Fe1Fe2O4@PEG-5000']['np_shell_1'] = 'PEG-5000'
my_df[my_df['np_core'] == 'Fe1Fe2O4@PEG-5000']['np_core'] = 'Fe1Fe2O4'

my_df[my_df['np_core'] == 'Eu1Fe1Fe2O4@PMAO-PEG']['np_shell_1'] = 'PMAO-PEG'
my_df[my_df['np_core'] == 'Eu1Fe1Fe2O4@PMAO-PEG']['np_core'] = 'Eu1Fe1Fe2O4'

my_df[my_df['np_core'] == 'Gd0.14Fe0.93Fe1.86O3.93@HDA-G2']['np_shell_1'] = 'HDA-G2'
my_df[my_df['np_core'] == 'Gd0.14Fe0.93Fe1.86O3.93@HDA-G2']['np_core'] = 'Gd0.14Fe0.93Fe1.86O3.93'

my_df[my_df['np_core'] == 'Gd2O3@HDA-G2']['np_shell_1'] = 'HDA-G2'
my_df[my_df['np_core'] == 'Gd2O3@HDA-G2']['np_core'] = 'Gd2O3'

my_df[my_df['np_core'] == 'Fe1Fe2O4@HDA-G2']['np_shell_1'] = 'HDA-G2'
my_df[my_df['np_core'] == 'Fe1Fe2O4@HDA-G2']['np_core'] = 'Fe1Fe2O4'

my_df[my_df['np_core'] == 'Dy1Si1O2@sulfo-SMCC']['np_shell_1'] = 'sulfo-SMCC'
my_df[my_df['np_core'] == 'Dy1Si1O2@sulfo-SMCC']['np_core'] = 'Dy1Si1O2'

my_df[my_df['np_core'] == 'Fe1Fe2O4@sulfo-SMCC']['np_shell_1'] = 'sulfo-SMCC'
my_df[my_df['np_core'] == 'Fe1Fe2O4@sulfo-SMCC']['np_core'] = 'Fe1Fe2O4'

my_df[my_df['np_core'] == 'Fe1Fe2O4@DA/FA']['np_shell_1'] = 'DA/FA'
my_df[my_df['np_core'] == 'Fe1Fe2O4@DA/FA']['np_core'] = 'Fe1Fe2O4'

my_df[my_df['np_core'] == 'Fe1Fe2O4@PEG-350']['np_shell_1'] = 'EG-350'
my_df[my_df['np_core'] == 'Fe1Fe2O4@PEG-350']['np_core'] = 'Fe1Fe2O4'

my_df[my_df['np_core'] == 'Fe1Fe2O4@PEG-1100']['np_shell_1'] = 'EG-1100'
my_df[my_df['np_core'] == 'Fe1Fe2O4@PEG-1100']['np_core'] = 'Fe1Fe2O4'

my_df[my_df['np_core'] == 'Fe1Pt1@DSPE-PEG5000-FA']['np_shell_1'] = 'DSPE'
my_df[my_df['np_core'] == 'Fe1Pt1@DSPE-PEG5000-FA']['np_shell_2'] = 'PEG5000-FA'
my_df[my_df['np_core'] == 'Fe1Pt1@DSPE-PEG5000-FA']['np_shell_3'] = 'FA'
my_df[my_df['np_core'] == 'Fe1Pt1@DSPE-PEG5000-FA']['np_core'] = 'Fe1Pt1'

my_df[my_df['np_core'] == 'Na1Gd1F4@PEG-mAb']['np_shell_1'] = 'PEG'
my_df[my_df['np_core'] == 'Na1Gd1F4@PEG-mAb']['np_shell_2'] = 'mAb'
my_df[my_df['np_core'] == 'Na1Gd1F4@PEG-mAb']['np_core'] = 'Na1Gd1F4'

my_df[my_df['np_core'] == 'Mn1O1@PEG-phospholipid']['np_shell_1'] = 'PEG'
my_df[my_df['np_core'] == 'Mn1O1@PEG-phospholipid']['np_shell_2'] = 'phospholipid'
my_df[my_df['np_core'] == 'Mn1O1@PEG-phospholipid']['np_core'] = 'Mn1O1'

my_df[my_df['np_core'] == 'Fe1Fe2O4@PEG-phospolipid']['np_shell_1'] = 'PEG'
my_df[my_df['np_core'] == 'Fe1Fe2O4@PEG-phospholipid']['np_shell_2'] = 'phospholipid'
my_df[my_df['np_core'] == 'Fe1Fe2O4@PEG-phospholipid']['np_core'] = 'Fe1Fe2O4'

my_df[my_df['np_core'] == 'Fe1Fe2O4@DSPE-PEG5k']['np_shell_1'] = 'DSPE'
my_df[my_df['np_core'] == 'Fe1Fe2O4@DSPE-PEG5k']['np_shell_2'] = 'PEG5k'
my_df[my_df['np_core'] == 'Fe1Fe2O4@DSPE-PEG5k']['np_core'] = 'Fe1Fe2O4'

my_df[my_df['np_core'] == 'Fe1Fe2O4@PEG-PMMA']['np_shell_1'] = 'PEG'
my_df[my_df['np_core'] == 'Fe1Fe2O4@PEG-PMMA']['np_shell_2'] = 'PMMA'
my_df[my_df['np_core'] == 'Fe1Fe2O4@PEG-PMMA']['np_core'] = 'Fe1Fe2O4'

my_df[my_df['np_core'] == 'Fe1Fe2O4@DP-PEG']['np_shell_1'] = 'DP'
my_df[my_df['np_core'] == 'Fe1Fe2O4@DP-PEG']['np_shell_2'] = 'PEG'
my_df[my_df['np_core'] == 'Fe1Fe2O4@DP-PEG']['np_core'] = 'Fe1Fe2O4'

my_df[my_df['np_core'] == 'Fe1Fe2O4@HX-PEG']['np_shell_1'] = 'HX'
my_df[my_df['np_core'] == 'Fe1Fe2O4@HX-PEG']['np_shell_2'] = 'PEG'
my_df[my_df['np_core'] == 'Fe1Fe2O4@HX-PEG']['np_core'] = 'Fe1Fe2O4'

my_df[my_df['np_core'] == 'Fe1Fe2O4@CC-PEG']['np_shell_1'] = 'CC'
my_df[my_df['np_core'] == 'Fe1Fe2O4@CC-PEG']['np_shell_2'] = 'PEG'
my_df[my_df['np_core'] == 'Fe1Fe2O4@CC-PEG']['np_core'] = 'Fe1Fe2O4'

my_df[my_df['np_core'] == 'Fe1Fe2O4@DSPE-mPEG1000']['np_shell_1'] = 'DSPE'
my_df[my_df['np_core'] == 'Fe1Fe2O4@DSPE-mPEG1000']['np_shell_2'] = 'mPEG1000'
my_df[my_df['np_core'] == 'Fe1Fe2O4@DSPE-mPEG1000']['np_core'] = 'Fe1Fe2O4'

my_df[my_df['np_core'] == 'Fe1Fe2O4@DSPE-mPEG550']['np_shell_1'] = 'DSPE'
my_df[my_df['np_core'] == 'Fe1Fe2O4@DSPE-mPEG550']['np_shell_2'] = 'mPEG550'
my_df[my_df['np_core'] == 'Fe1Fe2O4@DSPE-mPEG550']['np_core'] = 'Fe1Fe2O4'

my_df[my_df['np_core'] == 'Fe1Fe2O4@DSPE-mPEG750']['np_shell_1'] = 'DSPE'
my_df[my_df['np_core'] == 'Fe1Fe2O4@DSPE-mPEG750']['np_shell_2'] = 'mPEG750'
my_df[my_df['np_core'] == 'Fe1Fe2O4@DSPE-mPEG750']['np_core'] = 'Fe1Fe2O4'

my_df[my_df['np_core'] == 'Fe1Fe2O4@DSPE-mPEG2000']['np_shell_1'] = 'DSPE'
my_df[my_df['np_core'] == 'Fe1Fe2O4@DSPE-mPEG2000']['np_shell_2'] = 'mPEG2000'
my_df[my_df['np_core'] == 'Fe1Fe2O4@DSPE-mPEG2000']['np_core'] = 'Fe1Fe2O4'

my_df[my_df['np_core'] == 'Fe1Fe2O4@DSPE-mPEG5000']['np_shell_1'] = 'DSPE'
my_df[my_df['np_core'] == 'Fe1Fe2O4@DSPE-mPEG5000']['np_shell_2'] = 'mPEG5000'
my_df[my_df['np_core'] == 'Fe1Fe2O4@DSPE-mPEG5000']['np_core'] = 'Fe1Fe2O4'

my_df[my_df['np_core'] == 'Na1Dy1F4@PMAO-PEG']['np_shell_1'] = 'PMAO'
my_df[my_df['np_core'] == 'Na1Dy1F4@PMAO-PEG']['np_shell_2'] = 'PEG'
my_df[my_df['np_core'] == 'Na1Dy1F4@PMAO-PEG']['np_core'] = 'Na1Dy1F4'

my_df[my_df['np_core'] == 'Na1Ho1F4@DSPE-mPEG2000']['np_shell_1'] = 'DSPE'
my_df[my_df['np_core'] == 'Na1Ho1F4@DSPE-mPEG2000']['np_shell_2'] = 'mPEG2000'
my_df[my_df['np_core'] == 'Na1Ho1F4@DSPE-mPEG2000']['np_core'] = 'Na1Ho1F4'

my_df[my_df['np_core'] == 'Na1Ho1F4@PMAO-PEG']['np_shell_1'] = 'PMAO'
my_df[my_df['np_core'] == 'Na1Ho1F4@PMAO-PEG']['np_shell_2'] = 'PEG'
my_df[my_df['np_core'] == 'Na1Ho1F4@PMAO-PEG0']['np_core'] = 'Na1Ho1F4'

my_df[my_df['np_core'] == 'Mn1Fe2O4@PEG-PEI']['np_shell_1'] = 'PEG'
my_df[my_df['np_core'] == 'Mn1Fe2O4@PEG-PEI']['np_shell_2'] = 'PEI'
my_df[my_df['np_core'] == 'Mn1Fe2O4@PEG-PEI']['np_core'] = 'Mn1Fe2O4'

my_df[my_df['np_core'] == 'Mn1Fe2O4@Gallol-PEG']['np_shell_1'] = 'Gallol'
my_df[my_df['np_core'] == 'Mn1Fe2O4@Gallol-PEG']['np_shell_2'] = 'PEG'
my_df[my_df['np_core'] == 'Mn1Fe2O4@Gallol-PEG']['np_core'] = 'Mn1Fe2O4'

my_df[my_df['np_core'] == 'Fe1Fe2O4@NDOPA-PEG']['np_shell_1'] = 'NDOPA'
my_df[my_df['np_core'] == 'Fe1Fe2O4@NDOPA-PEG']['np_shell_2'] = 'PEG'
my_df[my_df['np_core'] == 'Fe1Fe2O4@NDOPA-PEG']['np_core'] = 'Fe1Fe2O4'

my_df[my_df['np_core'] == 'Mn1Fe2O4@NDOPA-PEG']['np_shell_1'] = 'NDOPA'
my_df[my_df['np_core'] == 'Mn1Fe2O4@NDOPA-PEG']['np_shell_2'] = 'PEG'
my_df[my_df['np_core'] == 'Mn1Fe2O4@NDOPA-PEG']['np_core'] = 'Mn1Fe2O4'

my_df[my_df['np_core'] == 'Zn0.2Mn0.8Fe2O4@NDOPA-PEG']['np_shell_1'] = 'NDOPA'
my_df[my_df['np_core'] == 'Zn0.2Mn0.8Fe2O4@NDOPA-PEG']['np_shell_2'] = 'PEG'
my_df[my_df['np_core'] == 'Zn0.2Mn0.8Fe2O4@NDOPA-PEG']['np_core'] = 'Zn0.2Mn0.8Fe2O4'


my_df['smiles_1'] = None
my_df['smiles_2'] = None
my_df['smiles_3'] = None

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
        [Composition(i) for i in nanodesclib.classes.assign_class(my_df.loc[j, 'np_shell_2']).consist()]
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

builder = DescriptorDatasetBuilder(dataframe=my_df, formula_col="np_core", smiles_cols=['smiles_1', 'smiles_2', 'smiles_3'])
result = builder.build()
result.to_csv('df_mri.csv', index=False)

preprocessor = DataPreprocessor(
    target_column="mri_r2",
    drop_nan_threshold=0.2,
    columns_to_drop=['paper_doi', 'paper_files', 'paper_comment', 'syn_doi', 'mri_h_val', 'mri_r1'],
    use_knn_imputer=True,
    encoding="target",
    scaling="none"
)
clean_df = preprocessor.fit_transform(result)

clean_df.to_csv('clean_df_mri.csv', index=False)