from nanodesclib.Calculation_to_dataframe import *
from nanodesclib.preprocess import *
import pubchempy as pcp

df = pd.read_csv(r'C:\Users\Public\Documents\nanomaterials_descriptors_library\tests\EDA\nanozymes.csv')

for i in df.index:
    for col in ['Km, mM', 'Vmax, mM/s', 'C max, mM', 'C(const), mM']:
        try:
            df.loc[i, col] = float((df.loc[i, col]).replace(',', ''))
        except:
            df.loc[i, col] = np.nan

df['Km, mM'] = df['Km, mM'].astype(float)
df['Vmax, mM/s'] = df['Vmax, mM/s'].astype(float)
df['C max, mM'] = df['C max, mM'].astype(float)
df['C(const), mM'] = df['C(const), mM'].astype(float)

link = list(df[df['Vmax, mM/s'] == max(df['Vmax, mM/s'])]['link'])[0]
drop_index = df[df['link'] == link].index
df = df.drop(drop_index, axis="index")

replace_dict_formula = {'4N-TiO2': 'N0.17-TiO2',
                        '(Co,Mn)3O4': 'Co3O4-Mn3O4',
                        'BNCuS': 'BN-CuS',
                        'NHMoO3': 'NH-MoO3'}
df['formula'] = df['formula'].replace(replace_dict_formula)

df['smiles_1'] = None

for i in df['pol'].index:
    try:
        df.loc[i, 'smiles_1'] = pcp.get_compounds(df.loc[i, 'np_shell_1'], 'name')[0].isomeric_smiles
    except:
        continue

builder = DescriptorDatasetBuilder(dataframe=df, formula_col="formula", smiles_cols=['smiles_1'])
result = builder.build()
result.to_csv('df_nz.csv', index=False)

preprocessor = DataPreprocessor(
    target_column="Km, mM",
    drop_nan_threshold=0.2,
    columns_to_drop=['#', 'link', 'Vmax, mM/s'],
    use_knn_imputer=True,
    encoding="target",
    scaling="none"
)
clean_df = preprocessor.fit_transform(result)

clean_df.to_csv('clean_df_nz_Km.csv', index=False)