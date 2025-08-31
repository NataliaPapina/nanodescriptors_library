from nanodesclib.Calculation_to_dataframe import *
from nanodesclib.preprocess import *

my_df = pd.read_csv(r'C:\Users\Public\Documents\nanomaterials_descriptors_library\tests\EDA\nanozymes.csv')

builder = DescriptorDatasetBuilder(dataframe=my_df.loc[:,['formula','Vmax, mM/s']], formula_col="formula")
result = builder.build()
result.to_csv('df_nz_Vm_only_lb.csv', index=False)

preprocessor = DataPreprocessor(
    target_column="Vmax, mM/s",
    drop_nan_threshold=0.2,
    #columns_to_drop=['#', 'link', 'Km, mM'],
    use_knn_imputer=True,
    encoding="target",
    scaling="minmax"
)
clean_df = preprocessor.fit_transform(result)

clean_df.to_csv('clean_df_nz_Vmax_only_lb.csv', index=False)