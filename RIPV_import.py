import numpy as np
from scipy import integrate, linalg
import qutip as q
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.linalg import expm
from tqdm import tqdm
from matplotlib.pyplot import cm
import pickle

def construct_H_c(coeffs, H_c_terms) -> tf.Tensor:
    """
    coeffs: Tensor[nCtrl, nTime]
    H_c_terms: Tensors[nCtrl, d, d]
    return: Tensor[nTime, d, d]
    """
    # Expand coeffs [nCtrl, nTime] → [nCtrl, nTime, 1, 1]
    coeffs_expanded = tf.expand_dims(tf.expand_dims(coeffs, axis=-1), axis=-1)

    # Expand H_c_terms [nCtrl, d, d] → [nCtrl, 1, d, d]
    H_c_terms_expanded = tf.expand_dims(H_c_terms, axis=1)

    # Element-wise multiplication, then summing over the control dimension
    H_c = tf.reduce_sum(coeffs_expanded * H_c_terms_expanded, axis=0)
    # H_c now has shape [nTime, d, d]
    
    return H_c


def propagator_of_Ht(Ht, tlist=None, dt=None) -> tf.Tensor:
    """
    - Ht: Tensor[nTime, d, d]s
    - optional parameters, choose either one:
        tlist: Tensor[nTime,]
        dt: scalar
    """
    assert (tlist is None) != (dt is None)  # one and only one is None
    nTime, d, _ = Ht.shape

    if tlist is not None:
        dt = tlist[1:] - tlist[:-1]  # shape=(nTime-1,)
        dt = tf.expand_dims(tf.expand_dims(dt, axis=1), axis=2)
    dU = tf.linalg.expm(-1j * Ht[:-1] * dt)

    # This is not cumulative product as U = tf.math.cumprod(dU, axis=0)
    # Instead, we do cumulative __matrix__ product
    Ut = [None] * nTime
    Ut[0] = tf.eye(d, dtype=Ht.dtype)
    for i in range(nTime - 1):
        Ut[i + 1] = dU[i] @ Ut[i]
    Ut = tf.convert_to_tensor(Ut)
    return Ut


def interaction_picture(H: tf.Tensor, U_0: tf.Tensor) -> tf.Tensor:
    """
    H: Tensor[nTime, d, d] or [d, d]
    U_0: Tensor[nTime, d, d]
    return: Tensor[nTime, d, d]
        H_I[k] = U_0[k]^† H[k] U_0[k]
    """
    if len(H.shape) == 2:  # H: Tensor[d, d] is time-independent
        H = tf.expand_dims(H, axis=0)  # H.shape: (d, d) -> (1, d, d)
    H_I = tf.matmul(U_0, H, adjoint_a=True) @ U_0
    return H_I


def simpleIntegral(y: tf.Tensor, *args, x=None, dx=None, axis=0) -> tf.Tensor:
    """
    Only support axis=0 now
    Definite integral
    y: Tensor[nTime, *shape]
    x: Tensor[nTime,]
    return: [*shape], one y value
        = ∫_x[0]^x[-1] y dx
    """
    assert axis == 0
    if x is not None:
        dx = x[1:] - x[:-1]
        # shape should be [nTimeSlices, 1(repeat d-1)] if y is rank-d tensor
        rank_y = len(y.shape)
        shape = (len(dx),) + (1,) * (rank_y - 1)
        dx = tf.reshape(dx, shape)
        if dx.dtype != tf.complex128:
            dx = tf.cast(dx, dtype=tf.complex128)
        I = tf.math.reduce_sum(y[:-1] * dx, axis=axis)
    elif dx is not None:
        # Assuming the pulse is 1d array
        I = tf.math.reduce_sum(y[:-1] * dx, axis=axis)
    else:
        raise Exception('Either x or dx must be passed with name')
    return I


def simpleCumulativeIntegral(yOfx, y0, x, axis=0) -> tf.Tensor:
    """
    Indefinite integral
    yOfx: Tensor[nTime, *y0.shape], values at corresponding x
    y0: initial value
    x: Tensor[nTime,]
    axis: only support axis=0
    return: Tensor[nTime, *y0.shape], array of y values
        = y0 + ∫_x[0]^x yOfx dx
    """
    assert axis == 0
    I = [None] * len(x)
    I[0] = y0
    dx = x[1:] - x[:-1]
    for i in range(len(dx)):
        # ∫_0^{x+dx} = ∫_0^x + y(x) * dx
        I[i + 1] = I[i] + yOfx[i] * dx[i]
    I = tf.stack(I)
    return I

def matrix_similarity(a, b, measure_type='Pederson fidelity'):
    if isinstance(a, q.Qobj):
        a = a.full()
    if isinstance(b, q.Qobj):
        b = b.full()
    def Pederson_fidelity(a, b):
        M = a.conj().T @ b
        Md = M.conj().T
        n = M.shape[0]
        return 1/(n*(n+1)) * (np.trace(M @ Md).real + np.abs(np.trace(M)) ** 2)
        
    type2func = {
        'inner product as infidelity':
            lambda a, b: (len(a) - np.trace(a.conj().T @ b)) / len(a),
        'tracedist':
            lambda a, b: q.tracedist(q.Qobj(a), q.Qobj(b)),
        'inner product':
            lambda a, b: np.abs(np.trace(a.conj().T @ b)) / len(a),
        'Pederson fidelity':
            Pederson_fidelity,
        'fro':
            lambda a, b: np.linalg.norm(a-b, 'fro'),
    }
    func = type2func[measure_type]
    d = func(a, b)
    return d