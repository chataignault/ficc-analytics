import numpy as np
from abc import ABC


class AbstractTraining(ABC):
    """
    Specify which method to use when optimizing portfolio weights
    """

    def __init__(self, x:np.ndarray):
        self._x = x
        self._trained = False

    @property
    def train(self)->bool:
        """
        Return whether the portfolio has been successfully optimized
        """
        return self._train

    def train(self):
        NotImplemented



class RobustCVTraining(AbstractTraining):
    """
    Robust cross-validation training of potfolio weights, 
    shuffling in-sample data to find market mode
    """

    def __init__(self, x):
        super().__init__(x)

    def train(self):
        return 


class CovarianceMatrixFilter(ABC):
    """
    Base class for covariance matrix filtering techniques
    """

    def __init__(self):
        pass

    def filter(self):
        NotImplemented


class ClippedCovarianceMatrixFilter(CovarianceMatrixFilter):
    NotImplemented
    

class RobustCVOracleTraining(AbstractTraining):
    """
    Robust cross-validation training of potfolio weights, 
    using oracle optimisation target, 
    """

    def __init__(self, x):
        super().__init__(x)

    def train(self):
        return 


class Portfolio(ABC):
    """
    Meta class to define a portfolio
    Focused on mid-frequency statistical arbitrage investment strategies
    """

    def __init__(self, ):
        pass

    def weight(self) -> np.ndarray:
        """
        Weight vector for a given strategy
        """
        NotImplemented


class MaxSharpePortfolio(Portfolio):
    
    def __init__(self,):
        super().__init__()

    def weight(self) -> np.ndarray:
        return 


class MinVarPortfolio(Portfolio):
    
    def __init__(self,):
        super().__init__()

    def weight(self) -> np.ndarray:
        return 


class MeanVarPortfolio(Portfolio):
    
    def __init__(self,):
        super().__init__()

    def weight(self) -> np.ndarray:
        return 
