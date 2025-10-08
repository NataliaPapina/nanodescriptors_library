from nanodesclib.AutoML import *

clean_df = pd.read_csv(r'C:\Users\Public\Documents\nanomaterials_descriptors_library\tests\preprocess_test\clean_df_nz_Km_only_lb.csv')
X = clean_df.drop(columns=["Km, mM", 'formula'])
y = clean_df["Km, mM"].apply(lambda x: np.log10(x+1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

automl = AutoMLPipeline(
    task="regression",
    pre_filter=True,
    #correlation_threshold=0.95,
    #var_threshold=0.01,
    n_trials=100,
    max_features=None,
    scale_data=False
)

automl.fit(X_train, y_train)

y_pred = automl.predict(X_test)

metrics = automl.score(X_test, y_test)
print("Метрики модели на тесте:", metrics)

print("Наиболее важные признаки:", automl.get_important_features())

automl.plot_shap_summary(X_test)