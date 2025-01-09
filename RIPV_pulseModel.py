import numpy as np
from scipy import integrate, linalg
import qutip as q
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.linalg import expm
from tqdm import tqdm
from matplotlib import cm
import matplotlib as mpl
import pickle

class HaiPulse:
    def __init__(self, tlist: tf.Tensor, envolope=tf.sin) -> None:
        self.tlist = tlist
        self.envolope = envolope
    
    def fromParams(self, p: tf.Tensor) -> tf.Tensor:
        """
        p: Tensor[2n+1,]
        where p[:n+1] are amplitudes
        p[n+1:] are phases
        """
        t = self.tlist
        T = t[-1] - t[0]
        n = (p.shape[0] - 1) // 2

        a, phi = p[:n+1], p[n+1:]
        j = tf.range(1.0, n+1, dtype=tf.float64)  # j = 1, ..., n

        t = tf.expand_dims(t, axis=0)       # t dim   [1, len_t]
        j = tf.expand_dims(j, axis=1)       # j dim   [n, 1]         where n = len_phi + len_a
        phi = tf.expand_dims(phi, axis=1)   # phi dim [n, 1] 
        a = tf.expand_dims(a, axis=1)       # a dim   [n + 1, 1]

        components = a[1:] * tf.cos(2 * np.pi * j * t / T + phi)

        pulse = (a[0] + tf.reduce_sum(components, axis=0)) * self.envolope(t / T * np.pi)
        pulse = tf.squeeze(pulse)
        pulse = tf.cast(pulse, tf.complex128)
        return pulse
    
    
class HaiTwoPulses(HaiPulse):
    def __init__(self, tlist: tf.Tensor, paramNumber: int, envolope=tf.sin) -> None:
        super().__init__(tlist, envolope=envolope)
        self.paramNumber = paramNumber
    
    def fromParams(self, p: tf.Tensor) -> tf.Tensor:
        halfL = self.paramNumber
        p1, p2 = p[:halfL], p[halfL:]
        pulse1 = super().fromParams(p1)
        pulse2 = super().fromParams(p2)
        return [pulse1, pulse2]

