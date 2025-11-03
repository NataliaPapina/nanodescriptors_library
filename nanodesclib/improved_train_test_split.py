from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

def improved_train_test_split(X, y, test_size=0.15, random_state=0):
    """Улучшенное разделение с проверкой статистик"""
    max_attempts = 10

    for attempt in range(max_attempts):
        current_seed = random_state + attempt

        n_bins = min(10, max(3, len(y) // 20))
        stratifier = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        y_binned = stratifier.fit_transform(y.values.reshape(-1, 1)).ravel()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=current_seed,
            stratify=y_binned
        )

        train_mean, test_mean = y_train.mean(), y_test.mean()
        train_std, test_std = y_train.std(), y_test.std()

        mean_diff = abs(train_mean - test_mean)
        std_diff = abs(train_std - test_std)

        if mean_diff < 0.008 and std_diff < 0.03:
            print(f"✅ Удачное разделение (попытка {attempt + 1})")
            break
    else:
        print("⚠️  Не удалось найти идеальное разделение, используем последнее")

    print("=== ПРОВЕРКА РАЗДЕЛЕНИЯ ===")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Train y: mean={y_train.mean():.3f}, std={y_train.std():.3f}")
    print(f"Test y:  mean={y_test.mean():.3f}, std={y_test.std():.3f}")
    print(f"Difference in means: {abs(y_train.mean() - y_test.mean()):.4f}")
    print(f"Difference in std: {abs(y_train.std() - y_test.std()):.4f}")

    return X_train, X_test, y_train, y_test