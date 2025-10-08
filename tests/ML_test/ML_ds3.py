from nanodesclib.AutoML import *

clean_df = pd.read_csv(r'C:\Users\Public\Documents\nanomaterials_descriptors_library\tests\preprocess_test\clean_df_ds3.csv')
X = clean_df.drop(columns=["Eg (eV) (Exp.)", "Metal oxide"], index=2)
y = clean_df["Eg (eV) (Exp.)"].drop(index=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

automl = AutoMLPipeline(
    enable_pls=True,
    task="regression",
    pre_filter=True,
    n_trials=100,
    max_features=None,
    scale_data=True
)

automl.fit(X_train, y_train)

y_pred = automl.predict(X_test)

metrics = automl.score(X_test, y_test)
print("Метрики модели на тесте:", metrics)

print("Наиболее важные признаки:", automl.get_important_features())

automl.plot_shap_summary(X_test)