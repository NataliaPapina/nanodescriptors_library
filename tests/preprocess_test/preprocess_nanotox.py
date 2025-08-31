from nanodesclib.Calculation_to_dataframe import *
from nanodesclib.preprocess import *

my_df = pd.read_csv(r'C:\Users\Public\Documents\nanomaterials_descriptors_library\tests\EDA\nanotox.csv')

builder = DescriptorDatasetBuilder(dataframe=my_df, formula_col="material")
result = builder.build()
result.to_csv('df_nt.csv', index=False)

preprocessor = DataPreprocessor(
    target_column="viability (%)",
    drop_nan_threshold=0.2,
    columns_to_drop=['DOI', 'Unnamed_0'],
    use_knn_imputer=True,
    encoding="target",
    scaling="minmax"
)
clean_df = preprocessor.fit_transform(result)

clean_df.to_csv('clean_df_nt.csv', index=False)