import numpy as np


def calculate_probabilities(data_arr, sex_col_data, target_col_data):
    """
    Q3: Tính xác suất cơ bản và xác suất có điều kiện.
    Giả sử sex_col_data chứa chuỗi 'M'/'F' hoặc số tương ứng, 
    target_col_data chứa 0 và 1.
    """
    # P(HeartDisease = 1)
    p_heart_disease = np.mean(target_col_data == 1)

    # P(HeartDisease = 1 | Sex = 'M')
    male_mask = (sex_col_data == 'M') | (sex_col_data == 1)  # hỗ trợ cả dạng chuỗi hoặc số mã hóa
    if np.sum(male_mask) == 0:
        p_hd_given_male = 0.0
    else:
        p_hd_given_male = np.mean(target_col_data[male_mask] == 1)

    return p_heart_disease, p_hd_given_male


def calculate_covariance_and_variance(data_arr, col_x, col_y):
    """
    Q5: Tính phương sai của col_x và hiệp phương sai giữa col_x và col_y
    """
    x = data_arr[:, col_x]
    y = data_arr[:, col_y]

    var_x = np.var(x, ddof=1)  # ddof=1 để tính toán theo mẫu (sample)
    cov_xy = np.cov(x, y, ddof=1)[0, 1]

    return var_x, cov_xy


def compute_covariance_eigen(data_arr):
    """
    Q6: Xây dựng ma trận hiệp phương sai và tính trị riêng, vectơ riêng
    """
    cov_matrix = np.cov(data_arr, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    return cov_matrix, eigenvalues, eigenvectors


def perform_svd(data_arr):
    """
    Q7: Thực hiện Singular Value Decomposition (SVD)
    """
    # X = U * Sigma * VT
    U, S, VT = np.linalg.svd(data_arr, full_matrices=False)
    return U, S, VT


def calculate_bayes_age(age_data, target_data):
    """
    Q9: Rời rạc hóa Age thành Young (<40), Middle-aged (40-55), Senior (>55)
    và áp dụng định lý Bayes tính P(HeartDisease=1 | Age_bin)
    """
    bins = []
    for age in age_data:
        if age < 40:
            bins.append('Young')
        elif age <= 55:
            bins.append('Middle-aged')
        else:
            bins.append('Senior')
    bins = np.array(bins)

    categories = ['Young', 'Middle-aged', 'Senior']
    bayes_results = {}

    p_hd = np.mean(target_data == 1)  # P(H)
    p_normal = np.mean(target_data == 0)  # P(N)

    for cat in categories:
        cat_mask = (bins == cat)
        if np.sum(cat_mask) == 0:
            bayes_results[cat] = 0.0
            continue

        # Tính toán theo định lý Bayes: P(H|C) = [P(C|H) * P(H)] / P(C)
        p_cat_given_hd = np.mean(bins[target_data == 1] == cat)  # P(C|H)
        p_cat_given_normal = np.mean(bins[target_data == 0] == cat)  # P(C|N)

        # Toàn phần P(C) = P(C|H)*P(H) + P(C|N)*P(N)
        p_cat = (p_cat_given_hd * p_hd) + (p_cat_given_normal * p_normal)

        p_hd_given_cat = (p_cat_given_hd * p_hd) / p_cat if p_cat > 0 else 0.0
        bayes_results[cat] = p_hd_given_cat

    return bins, bayes_results