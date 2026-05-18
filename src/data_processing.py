import numpy as np
import pandas as pd

# --- MODULE 1: NUMPY ---
def q1_load_data(file_path):
    df = pd.read_csv(file_path)
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    arr = df[numerical_cols].to_numpy()
    return arr, arr.shape, arr.dtype

def q2_statistics(arr):
    subset = arr[:, :3] # Age, RestingBP, Cholesterol
    return np.mean(subset, axis=0), np.median(subset, axis=0), np.std(subset, axis=0)

def q3_normalization(arr):
    return (arr - arr.min(axis=0)) / (arr.max(axis=0) - arr.min(axis=0))

def q10_batch_processing(arr, batch_size):
    n_batches = len(arr) // batch_size
    batches = np.array_split(arr[:n_batches*batch_size], n_batches)
    return [np.mean(batch, axis=0) for batch in batches]

# --- MODULE 3: PANDAS ---
def load_and_explore(file_path):
    """
    Q1: Load tập dữ liệu vào Pandas DataFrame và hiển thị cấu trúc cơ bản
    """
    df = pd.read_csv(file_path)
    print("--- 5 Dòng đầu của dữ liệu (head) ---")
    print(df.head())
    print("\n--- Thông tin cấu trúc dữ liệu (info) ---")
    df.info()
    print("\n--- Thống kê mô tả dữ liệu (describe) ---")
    print(df.describe(include='all'))  # include='all' để xem cả biến phân loại
    return df


def clean_heart_data(df):
    """
    Q2: Làm sạch dữ liệu. Xử lý các trường hợp Cholesterol hoặc RestingBP bằng 0.
    Thay thế bằng giá trị Trung bình (Mean) của cột đó (loại trừ các giá trị 0 cũ).
    """
    df_cleaned = df.copy()

    # Kiểm tra số lượng giá trị lỗi bằng 0 ban đầu
    zero_resting_bp = (df_cleaned['RestingBP'] == 0).sum()
    zero_cholesterol = (df_cleaned['Cholesterol'] == 0).sum()
    print(f"Số hàng có RestingBP = 0: {zero_resting_bp}")
    print(f"Số hàng có Cholesterol = 0: {zero_cholesterol}")

    # Thay thế 0 bằng NaN để không tính vào điểm trung bình (Mean)
    df_cleaned['RestingBP'] = df_cleaned['RestingBP'].replace(0, np.nan)
    df_cleaned['Cholesterol'] = df_cleaned['Cholesterol'].replace(0, np.nan)

    # Tính toán giá trị trung bình thực tế (bỏ qua NaN) và điền đầy (Imputation)
    mean_resting_bp = df_cleaned['RestingBP'].mean()
    mean_cholesterol = df_cleaned['Cholesterol'].mean()

    df_cleaned['RestingBP'] = df_cleaned['RestingBP'].fillna(mean_resting_bp)
    df_cleaned['Cholesterol'] = df_cleaned['Cholesterol'].fillna(mean_cholesterol)

    print("-> Đã làm sạch: Các giá trị lỗi bằng 0 đã được thay bằng giá trị trung bình tương ứng.")
    return df_cleaned