import numpy as np
from scipy import integrate, linalg
import qutip as q
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.linalg import expm
from tqdm import tqdm
from matplotlib.pyplot import cm
import pickle
from RIPV_import import *


class SingleControlModel:
    """
    This model defines how to control a system with pulse parameters.
    I.e. the following map:
        pulse parameters -> (rotation angle, robustness)
    """
    def __init__(self, tlist, H_sys, H_c_terms, H_n, paramsToSinglePulse) -> None:
        self.tlist = tlist
        self.dt = (tlist[-1] - tlist[0]) / len(tlist)
        self.H_sys = H_sys
        self.H_c_terms = H_c_terms
        self.H_n = H_n
        self.paramsToPulse = paramsToSinglePulse
        self.pulse = None
        self.H_I_t = None
        self.M1_t = None
        self.M2_T = None

    def setPulse(self, pulse):
        self.pulse = pulse

    def angleAndRobustness(self, A):
        """
        This function implements `CostAndConstraint`:
            f(A) -> {x, cost, cons, ...}
        """
        pulse = self.paramsToPulse(A)
        self.setPulse(pulse)
        theta = simpleIntegral(pulse, x=self.tlist)
        M1_t = self.get_M1_t()

        R1 = tf.norm(M1_t[-1])
        R1 = tf.expand_dims(R1, axis=0) # R1.shape = (nCons,)
        result = {
            'cost': theta,
            'cons': R1,
            'M1_t': M1_t,
            'H_I_t': self.H_I_t
        }
        return result

    def angleAndR1R2(self, A):
        pulse = self.paramsToPulse(A)
        self.setPulse(pulse)
        theta = simpleIntegral(pulse, x=self.tlist)
        M1_t = self.get_M1_t()

        R1 = tf.norm(M1_t[-1])
        M2_T = self.get_M2_T()
        R2 = tf.norm(M2_T)
        R1R2 = tf.stack([R1, R2])
        result = {
            'cost': theta,
            'cons': [R1, R2],
            'M1_t': M1_t,
            'H_I_t': self.H_I_t
        }
        return result
    
    def get_M1_t(self):
        assert self.pulse is not None
        pulse = self.pulse
        H_c = construct_H_c([pulse / 2], self.H_c_terms)
        H_noiseFree_t = self.H_sys + H_c
        U_noiseFree_t = propagator_of_Ht(H_noiseFree_t, dt=self.dt)

        H_I_t = interaction_picture(self.H_n, U_noiseFree_t)
        M1_t = simpleCumulativeIntegral(H_I_t, tf.convert_to_tensor(q.identity(2).full()), self.tlist)
        self.M1_t = M1_t
        self.H_I_t = H_I_t
        return M1_t
    
    def get_M2_T(self) -> tf.Tensor:
        """
        M2(t) = 1/2 ∫_0^t [H_I(t), M1(t)] dt
        """
        if self.M1_t is None:
            self.get_M1_t()
        H_I_t, M1_t = self.H_I_t, self.M1_t
        commutator = H_I_t @ M1_t - M1_t @ H_I_t
        M2_T = 1/4 * simpleIntegral(commutator, x=self.tlist, axis=0)
        self.M2_T = M2_T
        return M2_T
    

class AmplitudeModulatedMorlet:
    def __init__(self, tlist, halfWindowToSigmaRatio, nChannel):
        self.tlist = tlist
        self.nChannel = nChannel
        self.basis = self._buildBasis(halfWindowToSigmaRatio)
    
    def _buildBasis(self, halfWindowToSigmaRatio) -> tf.Tensor:
        """
        return: Tensor[nChannel, nTime]
        """
        t = self.tlist
        tstart, tend = t[0], t[-1]
        T = tend - tstart
        mu = (tstart + tend) / 2
        coses = [np.cos((k * 2 + 1) * (t - mu)/(T) * np.pi) for k in range(self.nChannel)]

        sigma = T/2 / halfWindowToSigmaRatio  # T/2 = 3sigma
        gaussWindow = np.exp(-1/2 * ((t - mu) / sigma)**2)
        cosesWithWindow = [cos * gaussWindow for cos in coses]
        return tf.convert_to_tensor(cosesWithWindow)

    def fromAmplitudes(self, A: tf.Tensor) -> tf.Tensor:
        """
        A: Tensor[nChannels]
        self.basis: Tensor[nChannels, nTime]
        return: Ω(t; A)
            A @ self.basis -> Tensor[nTime]
        """
        A = tf.expand_dims(A, axis=0)
        pulse = A @ self.basis
        pulse = tf.squeeze(pulse, axis=0)
        pulse = tf.cast(pulse, tf.complex128)
        return pulse

class XYControlModel:
    """
    This model assumes:
        control: X, Y
        noise:   X, Y, Z
    Hamiltonian:
        H = H_sys + H_c(t) + H_n
        H_c = ∑_i pulse_i * H_c_i
        H_n = ∑_i H_n_i
    """
    def __init__(self, tlist, paramsTo2Pulses, H_sys=0) -> None:
        X, Y, Z, I, OX = q.sigmax(), q.sigmay(), q.sigmaz(), q.identity(2), q.tensor
        X, Y, Z, I = [tf.constant(op.full()) for op in (X, Y, Z, I)]
        self.tlist = tlist
        self.dt = (tlist[-1] - tlist[0]) / len(tlist)
        self.H_sys = H_sys
        self.H_c_terms = [X, Y]
        self.H_n_terms = [X, Y, Z]
        self.paramsToPulses = paramsTo2Pulses

    def anglesAndRobs(self, A):
        """
        This function implements `costAndConstraint`:
            f(A) -> {x, cost, cons, ...}
        """
        X, Y, Z, I, OX = q.sigmax(), q.sigmay(), q.sigmaz(), q.identity(2), q.tensor
        X, Y, Z, I = [tf.constant(op.full()) for op in (X, Y, Z, I)]

        pulses = self.paramsToPulses(A)
        pulsesBy2 = [p/2 for p in pulses]
        H_c = construct_H_c(pulsesBy2, self.H_c_terms)
        
        H_sc_t = self.H_sys + H_c
        U_sc_t = propagator_of_Ht(H_sc_t, dt=self.dt)

        M1_t_list = []
        for H_n in self.H_n_terms:
            H_I_t = interaction_picture(H_n, U_sc_t)
            # M1_t = simpleCumulativeIntegral(H_I_t, I, self.tlist)
            M1 = simpleIntegral(H_I_t, x=self.tlist)
            M1_t = [M1]
            M1_t_list.append(M1_t)

        H_final = 1j * logm(U_sc_t[-1])
        # print(H_final, tf.linalg.expm(H_final))
        # H_final = 1j * tf.linalg.logm(U_sc_t[-1])
        # print(H_final, tf.linalg.expm(H_final))
        thetas = []
        for σ in (X, Y, Z):
            theta = tf.linalg.trace(tf.linalg.adjoint(σ) @ H_final)
            thetas.append(theta)
    
        R = [tf.norm(M1_t[-1]) for M1_t in M1_t_list]
        R = R + thetas[1:]  # R = [Rx, Ry, Rz, θy, θz]
        # R = tf.expand_dims(R, 0)
        result = {
            'cost': thetas[0],  # cost = θx
            'cons': R,
            # 'H_I_t': H_I_t,
            # 'M1_t': M1_t,
        }
        return result