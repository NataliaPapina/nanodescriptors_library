from nanodesclib.Calculation_to_dataframe import *
from nanodesclib.preprocess import *

my_df = pd.read_csv(r'C:\Users\Public\Documents\nanomaterials_descriptors_library\tests\EDA\nanozymes.csv')

builder = DescriptorDatasetBuilder(dataframe=my_df.loc[:,['formula', 'Km, mM']], formula_col="formula")
result = builder.build()
result.to_csv('df_nz_only_lb.csv', index=False)

preprocessor = DataPreprocessor(
    target_column="Km, mM",
    drop_nan_threshold=0.2,
    #columns_to_drop=['#', 'link', 'Vmax, mM/s'],
    use_knn_imputer=True,
    encoding="target",
    scaling="minmax"
)
clean_df = preprocessor.fit_transform(result)

clean_df.to_csv('clean_df_nz_Km_only_lb.csv', index=False)