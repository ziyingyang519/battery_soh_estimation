# 對 csv 操作
import csv
# 處理檔案路徑
from pathlib import Path
# numpy 用來數值計算
import numpy as np
# 匯入 MAE 和 MSE
from sklearn.metrics import mean_absolute_error, mean_squared_error
# 匯入不使用特徵的基準預測器
from sklearn.dummy import DummyRegressor
# 匯入特徵標準化工具
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
# 載入 matplotlib 的畫圖工具
import matplotlib.pyplot as plt 

# Final baseline: 4.0–4.2 V starts, first five seconds, IRWST only.
WINDOW_LABEL = "4p0_to_4p2"
DATA_SUFFIX = "_audit"
SPLIT_LABEL = "group_aware"
TRAIN_LAST_INTERVAL = 22
VALIDATION_LAST_INTERVAL = 30
SMOOTHING_WINDOWS = [1, 3, 5, 9, 15]
# 找到檔案路徑
project_root = Path(__file__).resolve().parents[1]
# 獎選取的特徵表位置，存到 data_path 這個變數
data_path = (
    project_root
    / "data"
    / "processed"
    / f"rw3_features_{WINDOW_LABEL}{DATA_SUFFIX}.csv"
)

feature_names = [
    "irwst_ohm",
]

# 讀取 csv 檔案，並將每一列存成字典，存到 rows 這個 list 裡面
with data_path.open(encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

def rows_to_arrays(split_rows):
    X = np.array(
        [
            [float(row[feature]) for feature in feature_names]
            for row in split_rows
        ],
        dtype=float,
    )
    y = np.array(
        [float(row["soh_percent"]) for row in split_rows],
        dtype=float,
    )
    return X, y


# Keep every reference-to-reference interpolation interval in one split.
train_rows = []
validation_rows = []
test_rows = []

for row in rows:
    interval_id = int(row["reference_interval_id"])

    if interval_id <= TRAIN_LAST_INTERVAL:
        train_rows.append(row)
    elif interval_id <= VALIDATION_LAST_INTERVAL:
        validation_rows.append(row)
    else:
        test_rows.append(row)

X_train, y_train = rows_to_arrays(train_rows)
X_validation, y_validation = rows_to_arrays(validation_rows)
X_test, y_test = rows_to_arrays(test_rows)

train_intervals = {
    int(row["reference_interval_id"])
    for row in train_rows
}
validation_intervals = {
    int(row["reference_interval_id"])
    for row in validation_rows
}
test_intervals = {
    int(row["reference_interval_id"])
    for row in test_rows
}

assert not train_intervals & validation_intervals
assert not train_intervals & test_intervals
assert not validation_intervals & test_intervals

print(f"Feature names: {feature_names}")
print(f"Total samples: {len(rows)}")
print(f"Train samples: {len(X_train)}")
print(f"Validation samples: {len(X_validation)}")
print(f"Test samples: {len(X_test)}")
print(
    f"Train reference intervals: "
    f"{min(train_intervals)} - {max(train_intervals)}"
)
print(
    f"Validation reference intervals: "
    f"{min(validation_intervals)} - {max(validation_intervals)}"
)
print(
    f"Test reference intervals: "
    f"{min(test_intervals)} - {max(test_intervals)}"
)
print(f"Train SOH range: {y_train.min():.2f}% - {y_train.max():.2f}%")
print(f"Test SOH range: {y_test.min():.2f}% - {y_test.max():.2f}%")

# 建立特徵標準化器 scaler，並對訓練集、驗證集、測試集進行標準化
scaler = StandardScaler()
# 計算訓練集的平均值和標準差，並將訓練集、驗證集、測試集進行標準化
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)
# 建立不使用特徵的 dummy baseline，以及主模型 Linear SVR。
models = {
    "Train-mean dummy": DummyRegressor(strategy="mean"),
    "Linear SVR": LinearSVR(
        # 設定懲罰參數 C，C 越大，模型對訓練資料的擬合能力越強，但可能會 overfit
        C=1.0,
        # 設定 epsilon，epsilon 越大，模型對誤差的容忍度越高，可能會 underfit
        epsilon=0.1,
        # 最多讓演算法迭代 20,000 次，避免太早停止。
        max_iter=20000,
        # 設定 random_state，讓每次訓練的結果都一樣，方便比較。
        random_state=42,
    ),
}

# 定義評估函數，計算 MAE、RMSE，並印出預測結果
def evaluate(model, split_name, X_split, y_true):
    y_pred = model.predict(X_split)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n{split_name}")
    print(f"MAE: {mae:.3f}%")
    print(f"RMSE: {rmse:.3f}%")
    print(
        f"Prediction range: "
        f"{y_pred.min():.2f}% - {y_pred.max():.2f}%"
    )

    return y_pred


def metric_values(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def print_prediction_metrics(label, y_true, y_pred):
    mae, rmse = metric_values(y_true, y_pred)
    print(f"{label}: MAE={mae:.3f}%, RMSE={rmse:.3f}%")
    return mae, rmse


def causal_moving_median(values, window_size):
    """Smooth each prediction using only the current and earlier predictions."""
    smoothed = np.empty_like(values)

    for index in range(len(values)):
        start_index = max(0, index - window_size + 1)
        smoothed[index] = np.median(values[start_index:index + 1])

    return smoothed


predictions = {}

for model_name, model in models.items():
    print(f"\n{'=' * 12} {model_name} {'=' * 12}")

    model.fit(X_train_scaled, y_train)

    predictions[model_name] = {
        "train": evaluate(model, "Train", X_train_scaled, y_train),
        "validation": evaluate(
            model,
            "Validation",
            X_validation_scaled,
            y_validation,
        ),
        "test": evaluate(model, "Test", X_test_scaled, y_test),
    }

# Select the smoothing window from RW3 validation only.
linear_predictions = predictions["Linear SVR"]
validation_history = np.concatenate(
    (linear_predictions["train"], linear_predictions["validation"])
)
validation_history_clipped = np.clip(validation_history, 0, 100)
validation_clipped = validation_history_clipped[len(y_train):]

print("\n============ Linear SVR post-processing selection ============")
print_prediction_metrics(
    "Validation raw",
    y_validation,
    linear_predictions["validation"],
)
print_prediction_metrics(
    "Validation clipped",
    y_validation,
    validation_clipped,
)

selected_window = None
best_validation_rmse = np.inf

for window_size in SMOOTHING_WINDOWS:
    validation_smoothed = causal_moving_median(
        validation_history_clipped,
        window_size,
    )[len(y_train):]
    _, validation_rmse = print_prediction_metrics(
        f"Validation clipped + causal median (window={window_size})",
        y_validation,
        validation_smoothed,
    )

    if validation_rmse < best_validation_rmse:
        best_validation_rmse = validation_rmse
        selected_window = window_size

print(f"Selected smoothing window from validation: {selected_window} samples")

# Test uses the already selected window and only earlier predictions as history.
test_history = np.concatenate(
    (
        linear_predictions["train"],
        linear_predictions["validation"],
        linear_predictions["test"],
    )
)
test_history_clipped = np.clip(test_history, 0, 100)
test_clipped = test_history_clipped[-len(y_test):]
test_smoothed = causal_moving_median(
    test_history_clipped,
    selected_window,
)[-len(y_test):]

print("\n============ Linear SVR final test post-processing ============")
print_prediction_metrics("Test raw", y_test, linear_predictions["test"])
print_prediction_metrics("Test clipped", y_test, test_clipped)
print_prediction_metrics(
    f"Test clipped + causal median (window={selected_window})",
    y_test,
    test_smoothed,
)

# Break the feature-label relationship in Train as a leakage sanity check.
rng = np.random.default_rng(42)
permuted_train_features = X_train_scaled[
    rng.permutation(len(X_train_scaled))
]
permuted_model = LinearSVR(
    C=1.0,
    epsilon=0.1,
    max_iter=20000,
    random_state=42,
)
permuted_model.fit(permuted_train_features, y_train)

print(f"\n{'=' * 12} Shuffled-IRWST sanity check {'=' * 12}")
evaluate(
    permuted_model,
    "Train (shuffled feature-label pairs)",
    permuted_train_features,
    y_train,
)
evaluate(
    permuted_model,
    "Validation",
    X_validation_scaled,
    y_validation,
)
evaluate(permuted_model, "Test", X_test_scaled, y_test)

test_order = np.arange(1, len(y_test) + 1)

plt.figure(figsize=(10, 5))
plt.plot(
    test_order,
    y_test,
    label="True SOH",
    color="black",
    linewidth=2,
)
plt.plot(
    test_order,
    predictions["Linear SVR"]["test"],
    label="Linear SVR raw",
)
plt.plot(
    test_order,
    test_smoothed,
    label=f"Linear SVR clipped + causal median (window={selected_window})",
)

plt.xlabel("Test sample order")
plt.ylabel("SOH (%)")
plt.title("RW3 chronological test: true vs predicted SOH")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

output_path = (
    project_root
    / "outputs"
    / f"svr_test_predictions_{WINDOW_LABEL}_{SPLIT_LABEL}.png"
)
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150)
plt.close()

print(f"\nSaved plot to: {output_path}")
