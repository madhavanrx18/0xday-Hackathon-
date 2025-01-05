rom typing import List, Union, Optional, Tuple, Callable
import tenseal as ts
from tenseal.tensors.abstract_tensor import AbstractTensor
import numpy as np
from functools import wraps


def requires_context(func):
    """Decorator to check if context is initialized before operations"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.context is None:
            raise ValueError("Operation requires an initialized context")
        return func(self, *args, **kwargs)
    return wrapper


class EnhancedCKKSVector(AbstractTensor):
    """Enhanced vector of values encrypted using CKKS scheme with additional features."""
    
    def _init_(
        self,
        context: "ts.Context" = None,
        vector: Union[List[float], np.ndarray] = None,
        scale: Optional[float] = None,
        data: "ts._ts_cpp.CKKSVector" = None,
    ):
        """Enhanced constructor with support for numpy arrays and additional validation."""
        self.context = context
        
        # wrapping existing data
        if data is not None:
            self.data = data
            return

        if not isinstance(context, ts.Context):
            raise TypeError("context must be a tenseal.Context")

        # Support for numpy arrays
        if isinstance(vector, np.ndarray):
            vector = vector.flatten().tolist()

        if not isinstance(vector, ts.PlainTensor):
            vector = ts.plain_tensor(vector, dtype="float")
            
        if len(vector.shape) != 1:
            raise ValueError("can only encrypt a vector")
            
        vector = vector.raw

        if scale is None:
            self.data = ts._ts_cpp.CKKSVector(context.data, vector)
        else:
            self.data = ts._ts_cpp.CKKSVector(context.data, vector, scale)

    # ... [Previous methods remain unchanged] ...

    @requires_context
    def exp(self) -> "EnhancedCKKSVector":
        """
        Approximate exponential function using Taylor series.
        Note: This is an approximation valid for small values.
        """
        result = self.clone()
        one = self._wrap(ts._ts_cpp.CKKSVector(self.context.data, [1.0] * self.size()))
        factorial = 1.0
        power = self.clone()
        
        # Add first 5 terms of Taylor series
        for i in range(1, 6):
            result = result.add(power.div(factorial))
            power = power.mul(self)
            factorial *= (i + 1)
        
        return result

    @requires_context
    def log(self) -> "EnhancedCKKSVector":
        """
        Approximate natural logarithm using Newton's method.
        Note: This is an approximation valid for values > 0.
        """
        # Initial guess
        result = self.clone()
        
        # Newton's method iterations
        for _ in range(5):
            exp_result = result.exp()
            result = result.sub(exp_result.sub(self).div(exp_result))
        
        return result

    @requires_context
    def sigmoid(self) -> "EnhancedCKKSVector":
        """
        Compute the sigmoid function (1 / (1 + e^(-x)))
        """
        neg = self.mul(-1)
        exp_neg = neg.exp()
        one = self._wrap(ts._ts_cpp.CKKSVector(self.context.data, [1.0] * self.size()))
        return one.div(one.add(exp_neg))

    @requires_context
    def tanh(self) -> "EnhancedCKKSVector":
        """
        Compute the hyperbolic tangent function
        """
        two = self._wrap(ts._ts_cpp.CKKSVector(self.context.data, [2.0] * self.size()))
        return two.mul(self.sigmoid()).sub(1.0)

    @requires_context
    def abs_approximate(self) -> "EnhancedCKKSVector":
        """
        Approximate absolute value using smooth approximation
        """
        epsilon = 1e-3
        return (self.mul(self).add(epsilon)).sqrt()

    @requires_context
    def clip(self, min_val: float, max_val: float) -> "EnhancedCKKSVector":
        """
        Clip values between min_val and max_val using smooth approximation
        """
        return self.max(min_val).min(max_val)

    @requires_context
    def max(self, threshold: float) -> "EnhancedCKKSVector":
        """
        Approximate maximum with threshold using smooth approximation
        """
        diff = self.sub(threshold)
        return threshold + diff.mul(diff.sigmoid())

    @requires_context
    def min(self, threshold: float) -> "EnhancedCKKSVector":
        """
        Approximate minimum with threshold using smooth approximation
        """
        diff = self.sub(threshold)
        return threshold + diff.mul(diff.sigmoid().sub(1))

    @requires_context
    def poly_eval(self, coefficients: List[float]) -> "EnhancedCKKSVector":
        """
        Evaluate a polynomial with given coefficients using Horner's method
        coefficients[0] + coefficients[1]x + coefficients[2]x^2 + ...
        """
        result = self._wrap(ts._ts_cpp.CKKSVector(self.context.data, 
                                                [coefficients[-1]] * self.size()))
        
        for coeff in reversed(coefficients[:-1]):
            result = result.mul(self).add(coeff)
            
        return result

    @requires_context
    def rolling_mean(self, window_size: int) -> "EnhancedCKKSVector":
        """
        Calculate rolling mean with specified window size
        """
        if window_size <= 0 or window_size > self.size():
            raise ValueError("Invalid window size")
            
        result = self.clone()
        for i in range(window_size):
            result = result.add(self.rotate(i))
        
        return result.div(float(window_size))

    @requires_context
    def apply_mask(self, mask: List[float]) -> "EnhancedCKKSVector":
        """
        Apply a binary mask to the vector
        """
        if len(mask) != self.size():
            raise ValueError("Mask size must match vector size")
            
        mask_vector = self._wrap(ts._ts_cpp.CKKSVector(self.context.data, mask))
        return self.mul(mask_vector)

    @requires_context
    def power(self, exponent: int) -> "EnhancedCKKSVector":
        """
        Raise vector elements to an integer power
        """
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError("Exponent must be a non-negative integer")
            
        result = self._wrap(ts._ts_cpp.CKKSVector(self.context.data, 
                                                [1.0] * self.size()))
        
        for _ in range(exponent):
            result = result.mul(self)
            
        return result

    @requires_context
    def reciprocal_approximate(self, iterations: int = 5) -> "EnhancedCKKSVector":
        """
        Approximate reciprocal (1/x) using Newton's method
        """
        # Initial guess
        result = self._wrap(ts._ts_cpp.CKKSVector(self.context.data, 
                                                [1.0] * self.size()))
        
        # Newton's method iterations
        for _ in range(iterations):
            result = result.mul(2.0 - self.mul(result))
            
        return result

    def get_noise_estimate(self) -> float:
        """
        Get an estimate of the noise level in the ciphertext
        """
        return self.data.noise_level() if hasattr(self.data, 'noise_level') else None

    @staticmethod
    def create_random(context: "ts.Context", size: int, 
                     min_val: float = 0.0, max_val: float = 1.0) -> "EnhancedCKKSVector":
        """
        Create a vector with random values between min_val and max_val
        """
        random_vector = np.random.uniform(min_val, max_val, size).tolist()
        return EnhancedCKKSVector(context=context, vector=random_vector)

    def compare_approximate(self, other, tolerance: float = 1e-3) -> "EnhancedCKKSVector":
        """
        Approximate comparison between two vectors using smooth approximation
        Returns values close to 1 where self > other, and close to 0 otherwise
        """
        diff = self.sub(other)
        return diff.div(tolerance).sigmoid()
