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


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt, alpha, zero = 4, 1., 0.5, 0
        # Create Kalman filter model matrices 1 constant velocity model.
        self._motion_mat1 = np.eye(3 * ndim, 3 * ndim)
        for i in range(ndim):
            self._motion_mat1[i, ndim + i] = dt
        for i in range(ndim):
            self._motion_mat1[2 * ndim + i, 2 * ndim + i] = zero
        self._update_mat1 = np.eye(2 * ndim, 3 * ndim)
        a = self._motion_mat1
        b = self._update_mat1
        # Create Kalman filter model matrices 2 constant acceleration model.
        self._motion_mat2 = np.eye(3 * ndim, 3 * ndim)
        c = self._motion_mat2
        for i in range(2 * ndim):
            self._motion_mat2[i, ndim + i] = dt
        for i in range(ndim):
            self._motion_mat2[i, 2 * ndim + i] = alpha
        self._update_mat2 = np.eye(2*ndim, 3 * ndim)
        d = self._update_mat2
        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity1 = 1. / 160
        self._std_weight_velocity2 = 1. / 20
        self._std_weight_acceleration = 1. / 160
        # self.imm_weightCV = 0.5
        # self.imm_weightCA = 0.5
        # self.imm_paraCT =       CT model weight
        self.imm_weight = np.array([0.1, 0.9])

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean_acc = np.zeros_like(mean_vel)
        mean_zero = np.zeros_like(mean_acc)
        mean1 = np.r_[mean_pos, mean_vel, mean_zero]
        mean2 = np.r_[mean_pos, mean_vel, mean_acc]

        std1 = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity1 * measurement[3],
            10 * self._std_weight_velocity1 * measurement[3],
            1e-5,
            10 * self._std_weight_velocity1 * measurement[3], 0, 0, 0, 0]

        std2 = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_velocity2 * measurement[3],
            2 * self._std_weight_velocity2 * measurement[3],
            1e-2,
            2 * self._std_weight_velocity2 * measurement[3],
            10 * self._std_weight_acceleration * measurement[3],
            10 * self._std_weight_acceleration * measurement[3],
            1e-5,
            10 * self._std_weight_acceleration * measurement[3]]
        covariance1 = np.diag(np.square(std1))
        covariance2 = np.diag(np.square(std2))
        mean = mean2
        # mean = self.imm_weight[0] * mean1 + self.imm_weight[1] * mean2
        # covariance = self.imm_weight[0] * covariance1 + self.imm_weight[1] * covariance2
        covariance = covariance2

        # covariance = self.imm_weight[0] * (covariance1 + (mean1 - mean) * (mean1 - mean).T) +\
        #              self.imm_weight[1] * (covariance2 + (mean2 - mean) * (mean2 - mean).T)
        return mean, covariance

    def predict(self, mean, covariance):

        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 12 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel1 = [
            self._std_weight_velocity1 * mean[3],
            self._std_weight_velocity1 * mean[3],
            1e-2,
            self._std_weight_velocity1 * mean[3]]
        std_vel2 = [
            self._std_weight_velocity2 * mean[3],
            self._std_weight_velocity2 * mean[3],
            1e-5,
            self._std_weight_velocity2 * mean[3]]
        std_acc = [
            self._std_weight_acceleration * mean[3],
            self._std_weight_acceleration * mean[3],
            1e-5,
            self._std_weight_acceleration * mean[3]]
        std_zero = [0, 0, 0, 0]
        motion_cov1 = np.diag(np.square(np.r_[std_pos, std_vel1, std_zero]))
        motion_cov2 = np.diag(np.square(np.r_[std_pos, std_vel2, std_acc]))

        # a = mean

        mean1 = np.dot(self._motion_mat1, mean)
        mean2 = np.dot(self._motion_mat2, mean)
        covariance1 = np.linalg.multi_dot((
            self._motion_mat1, covariance, self._motion_mat1.T)) + motion_cov1
        covariance2 = np.linalg.multi_dot((
            self._motion_mat2, covariance, self._motion_mat2.T)) + motion_cov2

        mean = self.imm_weight[0] * mean1 + self.imm_weight[1] * mean2
        covariance = self.imm_weight[0] * covariance1 + self.imm_weight[1] * covariance2
        # covariance = self.imm_weight[0] * (covariance1 + (mean1 - mean) * (mean1 - mean).T) + \
        #              self.imm_weight[1] * (covariance2 + (mean2 - mean) * (mean2 - mean).T)

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (12 dimensional array).
        covariance : ndarray
            The state's covariance matrix (12x12 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std1 = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3], 0, 0, 0, 0]

        std2 = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
            self._std_weight_velocity2 * mean[3],
            self._std_weight_velocity2 * mean[3],
            1e-1,
            self._std_weight_velocity2 * mean[3]]

        innovation_cov1 = np.diag(np.square(std1))
        innovation_cov2 = np.diag(np.square(std2))

        mean1 = np.dot(self._update_mat1, mean)
        mean2 = np.dot(self._update_mat2, mean)

        covariance1 = np.linalg.multi_dot((
            self._update_mat1, covariance, self._update_mat1.T))

        covariance2 = np.linalg.multi_dot((
            self._update_mat2, covariance, self._update_mat2.T))

        return mean1, covariance1 + innovation_cov1, mean2, covariance2 + innovation_cov2

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 8 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean1, projected_cov1, projected_mean2, projected_cov2 = self.project(
            mean, covariance)

        chol_factor1, lower = scipy.linalg.cho_factor(
            projected_cov1, lower=True, check_finite=False)
        chol_factor2, lower = scipy.linalg.cho_factor(
            projected_cov2, lower=True, check_finite=False)

        kalman_gain1 = scipy.linalg.cho_solve(
            (chol_factor1, lower), np.dot(covariance, self._update_mat2.T).T,
            check_finite=False).T
        kalman_gain2 = scipy.linalg.cho_solve(
            (chol_factor2, lower), np.dot(covariance, self._update_mat2.T).T,
            check_finite=False).T

        innovation1 = np.r_[measurement, np.zeros(4)] - projected_mean1
        innovation2 = np.r_[measurement, np.zeros(4)] - projected_mean2

        new_mean1 = mean + np.dot(innovation1, kalman_gain1.T)
        new_mean2 = mean + np.dot(innovation2, kalman_gain2.T)

        new_covariance1 = covariance - np.linalg.multi_dot((
            kalman_gain1, projected_cov1, kalman_gain1.T))
        new_covariance2 = covariance - np.linalg.multi_dot((
            kalman_gain2, projected_cov2, kalman_gain2.T))

        new_mean = self.imm_weight[0] * new_mean1 + self.imm_weight[1] * new_mean2
        new_covariance = self.imm_weight[0] * new_covariance1 + self.imm_weight[1] * new_covariance2
        # new_covariance = self.imm_weight[0] * (new_covariance1 + (new_mean1 - new_mean) * (new_mean1 - new_mean).T) + \
        #                  self.imm_weight[1] * (new_covariance2 + (new_mean2 - new_mean) * (new_mean2 - new_mean).T)

        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (12 dimensional).
        covariance : ndarray
            Covariance of the state distribution (12x12 dimensional).
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
        mean1, covariance1, mean2, covariance2 = self.project(mean, covariance)

        if only_position:
            mean1, covariance1 = mean1[:2], covariance1[:2, :2]
            measurements = measurements[:, :2]
        if only_position:
            mean2, covariance2 = mean2[:2], covariance2[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor1 = np.linalg.cholesky(covariance2)
        cholesky_factor2 = np.linalg.cholesky(covariance2)

        row = int( measurements.size / 4 )
        measurements = np.c_[measurements, np.zeros((row, 4))]
        d1 = measurements - mean1
        d2 = measurements - mean2

        z1 = scipy.linalg.solve_triangular(
            cholesky_factor1, d1.T, lower=True, check_finite=False,
            overwrite_b=True)
        z2 = scipy.linalg.solve_triangular(
            cholesky_factor2, d2.T, lower=True, check_finite=False,
            overwrite_b=True)

        squared_maha1 = np.sum(z1 * z1, axis=0)
        squared_maha2 = np.sum(z2 * z2, axis=0)
        squared_maha = self.imm_weight[0] * squared_maha1 + self.imm_weight[1] * squared_maha2

        return squared_maha
