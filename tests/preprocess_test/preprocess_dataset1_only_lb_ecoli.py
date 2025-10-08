import nanodesclib.classes
from nanodesclib.Calculation_to_dataframe import *
from nanodesclib.preprocess import *
import pubchempy as pcp
from pymatgen.core import Composition

my_df = pd.read_csv(r'C:\Users\Public\Documents\nanomaterials_descriptors_library\tests\dataset1.csv')

builder = DescriptorDatasetBuilder(dataframe=my_df.loc[:,['Metal oxides', 'E. coli log(1/EC50)']],
                                   formula_col="Metal oxides")
result = builder.build()
result.to_csv('df_ds1_only_lb.csv', index=False)

preprocessor = DataPreprocessor(
    target_column="E. coli log(1/EC50)",
    drop_nan_threshold=0.2,
    #columns_to_drop=['HacCrT log(1/LC50)'],
    use_knn_imputer=True,
    encoding="target",
    scaling="none"
)
clean_df = preprocessor.fit_transform(result)

clean_df.to_csv('clean_df_ds1_only_lb.csv', index=False)