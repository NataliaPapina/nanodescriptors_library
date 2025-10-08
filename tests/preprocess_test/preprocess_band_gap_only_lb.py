import nanodesclib.classes
from nanodesclib.Calculation_to_dataframe import *
from nanodesclib.preprocess import *
import pubchempy as pcp
from pymatgen.core import Composition

my_df = pd.read_csv(r'C:\Users\Public\Documents\nanomaterials_descriptors_library\tests\dataset3.csv', delimiter=',')
for i in my_df.index:
    my_df.loc[i, 'Metal oxide'] = my_df.loc[i, 'Metal oxide'].strip('-abr')

builder = DescriptorDatasetBuilder(dataframe=my_df.loc[:,['Metal oxide', 'Eg (eV) (Exp.)']], formula_col="Metal oxide")
result = builder.build()
result.to_csv('df_ds3_only_lb.csv', index=False)

preprocessor = DataPreprocessor(
    target_column="Eg (eV) (Exp.)",
    drop_nan_threshold=0.2,
    #columns_to_drop=['HacCrT log(1/LC50)'],
    use_knn_imputer=True,
    encoding="target",
    scaling="minmax"
)
clean_df = preprocessor.fit_transform(result)

clean_df.to_csv('clean_df_ds3_only_lb.csv', index=False)