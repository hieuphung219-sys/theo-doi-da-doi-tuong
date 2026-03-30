# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class UnscentedKalmanFilter:
    def __init__(self):
        ndim = 8
        self._ndim = ndim
        self._no_sigma_points = 2 * ndim + 1
        self._lamda = 3 - (ndim + 2)
        self._update_mat = np.eye(2, ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 100
        self._std_weight_acceleration = 1. / 100
        self.height = 0
        self.sigma_points = np.zeros((self._no_sigma_points + 4, self._ndim + 2))

    def initiate(self, measurement):
        mean_pos = measurement[:4] 
        mean_vel = np.zeros(4) 
        mean = np.r_[mean_pos, mean_vel] 

        std = [
            2 * self._std_weight_position * measurement[3], # x
            2 * self._std_weight_position * measurement[3], # y
            1e-2,                                           # a (tỷ lệ khung hình ít biến động)
            2 * self._std_weight_position * measurement[3], # h
            10 * self._std_weight_velocity * measurement[3], # vx
            10 * self._std_weight_velocity * measurement[3], # vy
            1e-2,
            10 * self._std_weight_velocity * measurement[3]  # vh
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def generate_sigma_point(self, mean, covariance):
        n = mean.shape[0]
        
        # --- VÁ LỖI POSITIVE DEFINITE ---
        # 1. Ép ma trận đối xứng để loại bỏ sai số lệch
        covariance = (covariance + covariance.T) / 2.0
        # 2. Thêm jitter (1e-4) vào đường chéo để đảm bảo luôn dương
        covariance += np.eye(n) * 1e-4
        # --------------------------------
        
        L = np.linalg.cholesky(covariance)
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = mean
        
        for i in range(n):
            sigma_points[i + 1] = mean + np.sqrt(n + self._lamda) * L[:, i]
            sigma_points[i + 1 + n] = mean - np.sqrt(n + self._lamda) * L[:, i]
            
        return sigma_points

    def augmentation(self, mean, covariance):
        mean_aug = np.zeros(10)
        mean_aug[:8] = mean
        covariance_aug = np.zeros((10, 10))
        covariance_aug[:8, :8] = covariance
        
        # Giữ nguyên phần tính toán std cho acceleration
        std = [
            self._std_weight_acceleration * self.height,
            1e-6
        ]
        covariance_aug[8:, 8:] = np.diag(np.square(std))
        return mean_aug, covariance_aug
    
    def predict(self, mean, covariance):
        mean, covariance = self.augmentation(mean, covariance)
        sigma_points = self.generate_sigma_point(mean, covariance)
        predicted_sigma_points = np.zeros((self._no_sigma_points + 4, self._ndim))
        
        for i in range(self._no_sigma_points + 4):
            x = sigma_points[i]
            # --- ÁNH XẠ INDEX MỚI (10 chiều = 8 trạng thái + 2 nhiễu) ---
            # x[0]=x, x[1]=y, x[2]=a, x[3]=h
            # x[4]=vx, x[5]=vy, x[6]=va, x[7]=vh
            # x[8]=acc_noise_x, x[9]=acc_noise_y
            
            dt = 1.0 # Giả định delta t = 1 frame
            
            # Cập nhật tọa độ x, y với nhiễu gia tốc mới ở x[8], x[9]
            x[0] += x[4] * dt + 0.5 * x[8] * dt**2 
            x[1] += x[5] * dt + 0.5 * x[9] * dt**2 
            
            # Cập nhật tỷ lệ khung hình (a) và chiều cao (h)
            x[2] += x[6] * dt # a = a + va*t (Hoạt động bình thường)
            x[3] += x[7] * dt # h = h + vh*t
            
            # CHÚ Ý: Phải lấy 8 phần tử trạng thái gốc thay vì 7 như trước
            predicted_sigma_points[i] = x[:8] 

        weights = np.zeros(self._no_sigma_points + 4)
        weights[0] = self._lamda / (self._ndim + 2 + self._lamda)
        weights[1:] = 0.5 / (self._ndim + 2 + self._lamda)
        mean = np.dot(weights, predicted_sigma_points)

        weights = np.diag(weights)
        covariance = np.linalg.multi_dot(
            ((mean.T - predicted_sigma_points).T, weights, (mean.T - predicted_sigma_points)))
        
        return mean, covariance, predicted_sigma_points

    def project(self, mean, covariance, height, predicted_sigma_points):
        # Sửa [:2] thành [:4] để lấy đủ x, y, a, h
        projected_sigma_points = predicted_sigma_points[:, :4].copy() 

        # Cập nhật số chiều cho weights
        weights = np.zeros(self._no_sigma_points + 4)
        weights[0] = self._lamda / (self._ndim + 2 + self._lamda)
        weights[1:] = 0.5 / (self._ndim + 2 + self._lamda)
        projected_mean = np.dot(weights, projected_sigma_points)

        weights_mat = np.diag(weights)
        delta = projected_sigma_points - projected_mean
        projected_covariance = np.linalg.multi_dot((delta.T, weights_mat, delta))
        
        # Thêm nhiễu đo lường cho cả 4 thành phần x, y, a, h
        std = [
            self._std_weight_position * height,
            self._std_weight_position * height,
            1e-1, # nhiễu cho aspect ratio
            self._std_weight_position * height
        ]
        innovation_cov = np.diag(np.square(std))
        
        # Tính toán Correlation (hiệp phương sai chéo)
        delta_x = predicted_sigma_points - mean
        correlation = np.linalg.multi_dot((delta_x.T, weights_mat, delta))
        
        return projected_mean, projected_covariance + innovation_cov, correlation

    def update(self, mean, covariance, measurement, predicted_sigma_points):
        print(f"[DEBUG UKF] Mean shape: {mean.shape} | Covariance shape: {covariance.shape}")
        projected_mean, projected_covariance, correlation = self.project(mean, covariance,
                                                                         measurement[3], predicted_sigma_points)
        kalman_gain = np.dot(correlation, np.linalg.inv(projected_covariance))

        innovation = measurement - projected_mean # measurement bây giờ có 4 phần tử

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_covariance, kalman_gain.T))
            
        # Tạm thời tắt hoặc sửa phần if check cũ vì nó đang dùng index cứng [:2]
        self.height = measurement[3]
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, height, predicted_sigma_points,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance, _ = self.project(mean, covariance, height, predicted_sigma_points)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
