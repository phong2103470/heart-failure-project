import numpy as np

class LinearRegressionCustom:
    def __init__(self):
        self.w = None
        self.b = None
        self.loss_history = []

    def fit_analytic(self, X, y):
        """
        Q2: Giải Linear Regression bằng công thức giải tích (Normal Equation)
        Để tính cả bias b, ta thêm một cột số 1 vào ma trận X.
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Thêm cột bias
        # w_ideal = (X_b^T * X_b)^(-1) * X_b^T * y
        w_optimal = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.b = w_optimal[0]
        self.w = w_optimal[1:]
        return self.w, self.b

    @staticmethod
    def compute_gradient(X, y, w, b):
        """
        Q4: Tính đạo hàm Loss MSE và trả về Gradient đối với w và b
        """
        n = X.shape[0]
        y_pred = X.dot(w) + b
        error = y_pred - y

        # Đạo hàm dL/dw = (2/n) * X^T * (Xw + b - y)
        dw = (2 / n) * X.T.dot(error)
        # Đạo hàm dL/db = (2/n) * sum(Xw + b - y)
        db = (2 / n) * np.sum(error)

        # Tính MSE Loss
        loss = np.mean(error ** 2)
        return dw, db, loss

    # def fit_gradient_descent(self, X, y, lr=0.00001, iterations=1000):
    #     """
    #     Q8: Triển khai thuật toán Gradient Descent để tối ưu w và b
    #     """
    #     n_features = X.shape[1]
    #     # Khởi tạo trọng số ngẫu nhiên ban đầu
    #     self.w = np.zeros(n_features)
    #     self.b = 0.0
    #     self.loss_history = []
    #
    #     for _ in range(iterations):
    #         dw, db, loss = self.compute_gradient(X, y, self.w, self.b)
    #
    #         # Cập nhật tham số ngược hướng Gradient
    #         self.w -= lr * dw
    #         self.b -= lr * db
    #         self.loss_history.append(loss)
    #
    #     return self.w, self.b, self.loss_history

    def fit_regularized(self, X, y, alpha=1.0, method='ridge'):
        """
        Q10: Áp dụng L1 (Lasso) hoặc L2 (Ridge) dựa trên giải thuật tối ưu phối hợp
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        n_features = X_b.shape[1]

        if method.lower() == 'ridge':
            # L2 Regularization: (X^T*X + alpha*I)^(-1) * X^T * y
            identity = np.eye(n_features)
            identity[0, 0] = 0  # Không điều hòa hệ số chặn bias
            w_optimal = np.linalg.inv(X_b.T.dot(X_b) + alpha * identity).dot(X_b.T).dot(y)
            self.b = w_optimal[0]
            self.w = w_optimal[1:]
        elif method.lower() == 'lasso':
            # Sử dụng Gradient Descent có phạt L1 để giải Lasso thủ công gọn nhẹ
            self.w = np.zeros(X.shape[1])
            self.b = 0.0
            lr = 0.00001
            for _ in range(2000):
                n = X.shape[0]
                y_pred = X.dot(self.w) + self.b
                error = y_pred - y
                # Phạt L1: thêm hệ số đạo hàm trị tuyệt đối sign(w)
                dw = (2 / n) * X.T.dot(error) + alpha * np.sign(self.w)
                db = (2 / n) * np.sum(error)
                self.w -= lr * dw
                self.b -= lr * db
        return self.w, self.b

    def fit_gradient_descent(self, X, y, lr=0.00002, iterations=150000):
        """
        Q8: Triển khai Gradient Descent ma trận kèm in tiến trình (Print Log)
        """
        n_samples = X.shape[0]
        X_b = np.c_[np.ones((n_samples, 1)), X]
        theta = np.zeros(X_b.shape[1])
        self.loss_history = []

        # Chạy vòng lặp (iterations + 1 để in được mốc cuối cùng)
        for i in range(iterations + 1):
            y_pred = X_b.dot(theta)
            error = y_pred - y
            loss = np.mean(error ** 2)

            # ĐIỀU CHỈNH: Cứ sau mỗi 10.000 vòng lặp hoặc ở vòng lặp đầu/cuối thì in kết quả ra
            if i % 10000 == 0 or i == iterations:
                b_current = theta[0]
                w1_current = theta[1]
                w2_current = theta[2]
                print(
                    f"Iteration {i:6d}: Loss = {loss:.6f} | w1 (Age) = {w1_current:.6f} | w2 (MaxHR) = {w2_current:.6f} | b = {b_current:.6f}")

            # Chỉ cập nhật trọng số nếu chưa đạt đến giới hạn vòng lặp cuối
            if i < iterations:
                gradient = (2 / n_samples) * X_b.T.dot(error)
                theta -= lr * gradient
                self.loss_history.append(loss)

        # Gán lại kết quả cuối cùng vào thuộc tính lớp
        self.b = theta[0]
        self.w = theta[1:]

        return self.w, self.b, self.loss_history