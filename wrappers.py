import numpy as np
import openturns as ot
from sklearn.base import BaseEstimator


class GpOTtoSklearnExpStd(BaseEstimator):
    """
    Standard-deviation conformal score for GP
    """
    def __init__(self, scale: int, amplitude: float, nu: float, power_std: float = 1) -> None:
        self.scale = scale
        self.amplitude = amplitude
        self.nu = nu
        self.trained_ = False
        self.power_std = power_std

    def fit(self, X_train, y_train):

        input_dim = X_train.shape[1]
        scale = input_dim * [self.scale]
        amplitude = [self.amplitude]

        covarianceModel = ot.MaternModel(scale, amplitude, self.nu)

        basis = ot.ConstantBasisFactory(input_dim).build()

        self.gp = ot.KrigingAlgorithm(ot.Sample(X_train), ot.Sample(y_train.reshape(-1, 1)), covarianceModel, basis)

        self.gp.run()

        self.trained_ = True

    def predict(self, X_test, return_std=False):

        metamodel = self.gp.getResult()(X_test)

        y_pred = metamodel.getMean()
        y_std = metamodel.getStandardDeviation() ** self.power_std

        if not return_std:
            return np.array(y_pred)
        else:
            return np.array(y_pred), np.array(np.exp(y_std))

    def __sklearn_is_fitted__(self):
        if self.trained_:
            return True
        else:
            return False


class GpOTtoSklearnStd(BaseEstimator):
    """
    Standard-deviation conformal score for GP
    """
    def __init__(self, scale: int, amplitude: float, nu: float, noise: float = None, power_std: float = 1) -> None:
        self.scale = scale
        self.amplitude = amplitude
        self.nu = nu
        self.trained_ = False
        self.noise = noise
        self.power_std = power_std

    def fit(self, X_train, y_train):

        input_dim = X_train.shape[1]
        scale = input_dim * [self.scale]
        amplitude = [self.amplitude]

        covarianceModel = ot.MaternModel(scale, amplitude, self.nu)

        basis = ot.ConstantBasisFactory(input_dim).build()

        self.gp = ot.KrigingAlgorithm(ot.Sample(X_train), ot.Sample(y_train.reshape(-1, 1)), covarianceModel, basis)
        if self.noise:
            np.random.seed(42)
            vec_noise = np.ones(len(X_train)) * self.noise
            self.gp.setNoise(vec_noise)
        self.gp.run()

        self.trained_ = True

    def predict(self, X_test, return_std=False):

        metamodel = self.gp.getResult()(X_test)

        y_pred = metamodel.getMean()
        y_std = metamodel.getStandardDeviation()

        if not return_std:
            return np.array(y_pred)
        else:
            return np.array(y_pred), np.array(y_std) ** self.power_std

    def __sklearn_is_fitted__(self):
        if self.trained_:
            return True
        else:
            return False


class BaggedGP(BaseEstimator):
    """
    Standard-deviation conformal score for GP
    """
    def __init__(
            self, n_estimators, scale: int, amplitude: float,
            nu: float, noise: float = None, power_std: float = 1
    ) -> None:
        self.n_estimators = n_estimators
        self.scale = scale
        self.amplitude = amplitude
        self.nu = nu
        self.trained_ = False
        self.noise = noise
        self.power_std = power_std

    def _initiate_estimators(self):
        estimators = []
        for _ in range(self.n_estimators):
            estimators.append(
                GpOTtoSklearnStd(
                    scale=self.scale, amplitude=self.amplitude,
                    nu=self.nu, noise=self.noise, power_std=self.power_std
                )
            )
        return estimators

    def _initiate_databags(self, X_train, y_train):
        databags = []
        n_points = len(X_train)
        for i in range(self.n_estimators):
            indices = np.random.choice(n_points, n_points // 2, replace=False)
            databags.append((X_train[indices], y_train[indices]))

        return databags

    def fit(self, X_train, y_train):
        self.estimators = self._initiate_estimators()
        databags = self._initiate_databags(X_train, y_train)

        for i in range(self.n_estimators):
            X, y = databags[i]
            self.estimators[i].fit(X, y)

    def predict(self, X_test, return_std=False):
        preds, std = [], []
        for i in range(self.n_estimators):
            preds_temp, std_temp = self.estimators[i].predict(X_test, return_std=True)
            preds.append(preds_temp)
            std.append(std_temp)

        if not return_std:
            return np.array(preds).mean(axis=0)
        else:
            return np.array(preds).mean(axis=0), np.array(std).mean(axis=0)

    def __sklearn_is_fitted__(self):
        if self.trained_:
            return True
        else:
            return False
