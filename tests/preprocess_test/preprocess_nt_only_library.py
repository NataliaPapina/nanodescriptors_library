from nanodesclib.Calculation_to_dataframe import *
from nanodesclib.preprocess import *

my_df = pd.read_csv(r'C:\Users\Public\Documents\nanomaterials_descriptors_library\tests\EDA\nanotox.csv')

builder = DescriptorDatasetBuilder(dataframe=my_df.loc[:,["material", "viability (%)"]], formula_col="material")
result = builder.build()
result.to_csv('df_nt_only_library.csv', index=False)

preprocessor = DataPreprocessor(
    target_column="viability (%)",
    drop_nan_threshold=0.2,
    use_knn_imputer=True,
    encoding="onehot",
    scaling="none"
)
clean_df = preprocessor.fit_transform(result)

clean_df.to_csv('clean_df_nt_only_library.csv', index=False)