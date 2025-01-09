import numpy as np
from scipy import integrate, linalg
import qutip as q
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.linalg import expm
from tqdm import tqdm
from matplotlib.pyplot import cm
import multiprocessing
import pickle

def innerProduct(a: tf.Tensor, b: tf.Tensor):
    """
    a and b are rank-1 Tensors
    """
    assert len(a.shape) == 1 and len(b.shape) == 1
    c = tf.reduce_sum(tf.math.conj(a) * b, axis=0)
    return c

def orthogonalizeTo(v, perpToW):
    """
    Assuming
    v: Tensor[L,]
    w: Tensor[L,]
    """
    w = perpToW
    v = tf.expand_dims(v, axis=1)  # column vector
    w = tf.expand_dims(w, axis=1)  # column vector
    # v = v - (v @ w) / (w @ w) * w
    v = v - tf.matmul(v, w, adjoint_a=True) / tf.matmul(w, w, adjoint_a=True) * w
    v = tf.squeeze(v, axis=1)
    return v

# This function is compatible with one `w` parameter of shape [1, L]
def orthogonalizeToMultiple(v, perpToW_s):
    """
    Assuming each vector has length L
    v: Tensor[L,]
    ws: List[Tensor[L], size=nW]
    """
    orthonormalBasis = []
    for w in perpToW_s:
        # Make each w into the basis, assuming all w lin. independent
        ortho = w
        for b in orthonormalBasis:
            ortho = ortho - innerProduct(w, b) * b
        orthonormal = ortho / tf.sqrt(innerProduct(ortho, ortho))
        orthonormalBasis.append(orthonormal)
    vPerp = v
    for w in orthonormalBasis:
        vPerp = vPerp - innerProduct(v, w) * w
    return vPerp


class FrequentSaver:
    def __init__(self, folder, name, saveFreq=10):
        self.name1 = f'{folder}/{name}B.npz'
        self.name2 = f'{folder}/{name}A.npz'
        self.saveFreq = saveFreq

    def save(self, data: dict):
        if data['itersCount'] // self.saveFreq % 2 == 1:
            filename = self.name1
        else:
            filename = self.name2
        with open(filename, mode='wb') as f:
            pickle.dump(data, f)
        print(f'{data["itersCount"]} iterations. Saved at {filename}')
        return

def gradientOrthogonalVariation(x0, costAndConstraint, iters=None, costStep=None, deviceID='/GPU:0'):
    """
    Optimize `cost` while keeping `constraint` unchanged.
    x0: initial value that satisfies constraint condition.
    costAndConstraint: function of the form
        x -> result: dict{'cost', 'cons', **kwargs}
            containing 1 cost, nCons constraints
            result['cost']: Tensor[,] (scalar)
            result['cons']: Tensor[nCons,]
    """
    from collections import defaultdict
    result = defaultdict(lambda: [])

    saver = FrequentSaver('./data/XYcontrolXYZnoise', 'result', saveFreq=10)

    with tf.device(deviceID):
        x = tf.Variable(x0)
        
        for iteration in tqdm(range(iters)):
            with tf.GradientTape(persistent=True) as tape:
                thisResult = costAndConstraint(x)

            result['x'].append(x.read_value())
            for k, v in thisResult.items():
                result[k].append(v)
            if iteration % 10 == 0:
                result['itersCount'] = iteration
                saver.save(dict(result))
                # multiprocessing.Process(target=saver.save, args=(dict(result)),).start()

            J = thisResult['cost']
            Rs = thisResult['cons']
            dJdx = tape.gradient(J, x)  # shape=(len(x),)
            dRdx = [tape.gradient(Ri, x) for Ri in Rs]  # shape=(nCons, len(x))
            _dx = orthogonalizeToMultiple(dJdx, dRdx)  # shape=(len(x),)
            alpha = tf.squeeze(costStep / innerProduct(dJdx, _dx))  # scalar
            dx = alpha * _dx

            x.assign_sub(dx)

    lastResult = costAndConstraint(x)
    result['x'].append(x.read_value())
    for k, v in lastResult.items():
        result[k].append(v)
    result['itersCount'] = iters
    result = dict(result)
    saver.save(result)
    return result

def gradientOrthogonalVariation_v1(x0, costAndConstraint, iters=None, costStep=None, deviceID='/GPU:0'):
    """
    Fixed: cosine missing in normalization of dA

    Optimize `cost` while keeping `constraint` unchanged.
    x0: initial value that satisfies constraint condition.
    costAndConstraint: function of the form
        x -> result: dict{'cost', 'cons', **kwargs}
            containing 1 cost, nCons constraints
            result['cost']: Tensor[,] (scalar)
            result['cons']: Tensor[nCons,]
    """
    from collections import defaultdict
    result = defaultdict(lambda: [])

    saver = FrequentSaver('./data/XYcontrolXYZnoise', 'result', saveFreq=10)

    with tf.device(deviceID):
        x = tf.Variable(x0)
        
        for iteration in tqdm(range(iters)):
            with tf.GradientTape(persistent=True) as tape:
                thisResult = costAndConstraint(x)

            result['x'].append(x.read_value())
            for k, v in thisResult.items():
                result[k].append(v)
            if iteration % 10 == 0:
                result['itersCount'] = iteration
                saver.save(dict(result))
                # multiprocessing.Process(target=saver.save, args=(dict(result)),).start()

            J = thisResult['cost']
            Rs = thisResult['cons']
            dJdx = tape.gradient(J, x)  # shape=(len(x),)
            dRdx = [tape.gradient(Ri, x) for Ri in Rs]  # shape=(nCons, len(x))
            dx_perp = orthogonalizeToMultiple(dJdx, dRdx)  # shape=(len(x),)
            alpha = tf.squeeze(costStep / innerProduct(dJdx, dx_perp))  # scalar
            dx = alpha * dx_perp

            x.assign_sub(dx)

    lastResult = costAndConstraint(x)
    result['x'].append(x.read_value())
    for k, v in lastResult.items():
        result[k].append(v)
    result['itersCount'] = iters
    result = dict(result)
    saver.save(result)
    return result

