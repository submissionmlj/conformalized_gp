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
