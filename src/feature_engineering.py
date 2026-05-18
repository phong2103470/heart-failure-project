import pandas as pd


def analyze_chest_pain(df):
    """
    Q3: Gom nhóm dữ liệu theo ChestPainType, tính tổng số bệnh nhân và tỷ lệ mắc bệnh tim
    """
    grouped = df.groupby('ChestPainType').agg(
        Total_Patients=('HeartDisease', 'count'),
        Heart_Disease_Rate=('HeartDisease', 'mean')
    ).reset_index()
    return grouped


def create_ratio_feature(df):
    """
    Q4: Tạo thuộc tính mới MaxHR_to_Age_Ratio = MaxHR / Age
    """
    df_new = df.copy()
    df_new['MaxHR_to_Age_Ratio'] = df_new['MaxHR'] / df_new['Age']
    return df_new


def create_pivot_ecg(df):
    """
    Q6: Tạo bảng xoay pivot table với RestingECG là index, HeartDisease làm cột
    """
    pivot = df.pivot_table(
        index='RestingECG',
        columns='HeartDisease',
        values='Age',  # Dùng cột bất kỳ để đếm số lượng
        aggfunc='count',
        fill_value=0
    )
    return pivot


def bin_age_groups(df):
    """
    Q7: Phân nhóm tuổi thành các phân khúc (<40, 40-55, 55+) và phân tích tỷ lệ bệnh
    """
    df_new = df.copy()
    # Định nghĩa các khoảng cắt và nhãn tương ứng
    bins = [0, 39, 55, 120]
    labels = ['<40', '40-55', '55+']
    df_new['Age_Group'] = pd.cut(df_new['Age'], bins=bins, labels=labels)

    analysis = df_new.groupby('Age_Group', observed=False).agg(
        Total_Patients=('HeartDisease', 'count'),
        Heart_Disease_Rate=('HeartDisease', 'mean')
    ).reset_index()
    return df_new, analysis


def bin_cholesterol_groups(df):
    """
    Q8: Phân nhóm Cholesterol (Normal, Borderline, High) và phân tích hành vi bệnh
    """
    df_new = df.copy()
    # Theo chuẩn y khoa cơ bản: Normal (<200), Borderline (200-239), High (>=240)
    bins = [0, 199, 239, 1000]
    labels = ['Normal', 'Borderline', 'High']
    df_new['Cholesterol_Group'] = pd.cut(df_new['Cholesterol'], bins=bins, labels=labels)

    analysis = df_new.groupby('Cholesterol_Group', observed=False).agg(
        Total_Patients=('HeartDisease', 'count'),
        Heart_Disease_Rate=('HeartDisease', 'mean')
    ).reset_index()
    return df_new, analysis


def perform_one_hot_encoding(df):
    """
    Q10: Thực hiện One-Hot Encoding cho các biến phân loại để chuẩn bị dữ liệu cho ML
    """
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    # Sử dụng pd.get_dummies, ép kiểu sang int (0/1) để ma trận dữ liệu sạch sẽ
    df_encoded = pd.get_dummies(df, columns=categorical_cols, dtype=int)
    return df_encoded