import numpy as np
import os
from cvxopt import solvers, matrix
from munkres import Munkres
from scipy.special import softmax


def power_iteration(A, num_simulations: int):
    """
    Compute principle eigen-vector of matrix A
    :param A:
    :param num_simulations:
    :return:
    """
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_simulations):
        # Calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)

        # Calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # Re normalize the vector
        b_k = b_k1 / b_k1_norm
    return b_k


class GraphMatchingWrapper(object):
    @staticmethod
    def generate_K_v1(deep_distance, buffer_ids1, buffer_ids2, coms1, coms2):
        """
        Generate affinity matrix for edges (related position) and nodes
        :param deep_distance: Euclidean distance between ResNet34 latent vector
        :param buffer_ids1: ID of reference components
        :param buffer_ids2: ID of target components
        :param coms1: dict of reference components
        :param coms2: dict of target components
        :return:
        """
        Nr = len(buffer_ids1)
        Nt = len(buffer_ids2)
        N_max = max(Nr, Nt)

        # Get centers
        center1 = np.array([coms1[i]['centroid'] for i in buffer_ids1])
        center2 = np.array([coms2[i]['centroid'] for i in buffer_ids2])

        area1 = np.array([coms1[i]['area'] for i in buffer_ids1])
        area1 = np.expand_dims(area1, -1)
        area2 = np.array([coms2[i]['area'] for i in buffer_ids2])
        area2 = np.expand_dims(area2, -1)

        # Process dummy
        N_dummy = abs(Nr - Nt)
        if Nr > Nt:
            center2 = np.vstack((center2, np.NaN * np.ones((N_dummy, 2))))
            area2 = np.vstack((area2, np.NaN * np.ones((N_dummy, 1))))
        elif Nr < Nt:
            center1 = np.vstack((center1, np.NaN * np.ones((N_dummy, 2))))
            area1 = np.vstack((area1, np.NaN * np.ones((N_dummy, 1))))

        # Distance
        c1 = np.expand_dims(center1, 0)
        c2 = np.expand_dims(center1, 1)
        distance1 = np.linalg.norm(c2 - c1, ord=2, axis=-1)

        c1 = np.expand_dims(center2, 0)
        c2 = np.expand_dims(center2, 1)
        distance2 = np.linalg.norm(c2 - c1, ord=2, axis=-1)

        distance = np.zeros((N_max ** 2, N_max ** 2))
        el = 0
        for i in range(N_max):
            for j in range(N_max):
                a_ = distance2[i, :]
                b_ = distance1[j, :]
                a_ = np.expand_dims(a_, 1)
                b_ = np.expand_dims(b_, 0)
                c = np.abs(a_ - b_) / (np.maximum(a_, b_) + 1e-4)
                distance[el, :] = c.flatten()
                el += 1

        # # Cosine
        # c1 = np.expand_dims(center1, 0)
        # c2 = np.expand_dims(center1, 1)
        # vector1 = (c2 - c1)
        #
        # c1 = np.expand_dims(center2, 0)
        # c2 = np.expand_dims(center2, 1)
        # vector2 = (c2 - c1)
        #
        # cosine = np.zeros((N_max ** 2, N_max ** 2))
        # el = 0
        # for i in range(N_max):
        #     for j in range(N_max):
        #         a_ = vector2[i, :, :]
        #         b_ = vector1[j, :, :]
        #         a_ = np.expand_dims(a_, 1)
        #         b_ = np.expand_dims(b_, 0)
        #         c = np.sum(a_ * b_, axis=-1)
        #         den = np.linalg.norm(a_, ord=2, axis=-1) * np.linalg.norm(b_, ord=2, axis=-1)
        #         c = 1 - (c / (den + 1e-5))
        #
        #         cosine[el, :] = c.flatten()
        #         el += 1

        # Area
        a1 = np.expand_dims(area1, 0)
        a2 = np.expand_dims(area1, 1)
        area1_ = (a2 * a1)[:, :, 0]

        a1 = np.expand_dims(area2, 0)
        a2 = np.expand_dims(area2, 1)
        area2_ = (a2 * a1)[:, :, 0]

        area = np.zeros((N_max ** 2, N_max ** 2))
        el = 0
        for i in range(N_max):
            for j in range(N_max):
                a_ = area2_[i, :]
                b_ = area1_[j, :]
                a_ = np.expand_dims(a_, 1)
                b_ = np.expand_dims(b_, 0)
                c = np.abs(a_ - b_) / (np.maximum(a_, b_))
                area[el, :] = c.flatten()
                el += 1

        # Label
        node_r = np.zeros((1, N_max ** 2), dtype=np.int)
        node_t = np.zeros((1, N_max ** 2), dtype=np.int)

        el = 0
        for t in range(N_max):
            for r in range(N_max):
                node_r[0, el] = r
                node_t[0, el] = t
                el += 1

        # Add deep feature to matrix K
        buffer_deep_distance = np.ones((Nt, Nr))
        for i in range(Nt):
            for j in range(Nr):
                buffer_deep_distance[i, j] = deep_distance[buffer_ids2[i], buffer_ids1[j]]
        if Nr > Nt:
            buffer_deep_distance = np.vstack((buffer_deep_distance, np.ones((N_dummy, Nr))))
        elif Nr < Nt:
            buffer_deep_distance = np.hstack((buffer_deep_distance, np.ones((Nt, N_dummy))))
        buffer_deep_distance = buffer_deep_distance.flatten()

        K = area

        # Add deep feature for K(a, a)
        el = 0
        for i in range(N_max ** 2):
            tmp = K[i, i]
            K[i, :] = np.exp((- K[i, :] - 4 * distance[i, :] - buffer_deep_distance[el]) / 3)
            K[i, i] = np.exp((- tmp - 2 * buffer_deep_distance[el]) / 2)
            el += 1

        nan_ids = np.where(np.isnan(K))
        K[nan_ids[0], nan_ids[1]] = 0
        return K, node_r, node_t

    @staticmethod
    def spectral_matching(M, node_r, node_t, buffer_ids1, buffer_ids2, min_ids):
        """
        Find solution for quadratic form
        :param M: Affinity matrix
        :param node_r:
        :param node_t:
        :param buffer_ids1:
        :param buffer_ids2:
        :param min_ids: correspondence ids from the first phase
        :return: min_ids after applying correspondence of phase 2
        """
        eps = 1e-8
        v = np.ones((node_t.shape[1], 1))
        v = v / np.linalg.norm(v)

        iterClimb = 30
        nNodes = np.max(node_t) + 1
        nLabels = np.max(node_r) + 1

        for i in range(iterClimb):
            v = np.dot(M, v)
            v = v / np.linalg.norm(v)

        aux = v
        v0 = aux
        v1 = aux

        for k in range(10):
            for j in range(nNodes):
                f = np.where(node_t == j)
                v1[f[1]] = v0[f[1]] / (np.sum(v0[f[1]]) + eps)

            for j in range(nLabels):
                f = np.where(node_r == j)
                v0[f[1]] = v1[f[1]] / (np.sum(v1[f[1]]) + eps)

        v = (v1 + v0) / 2
        v = v / np.linalg.norm(v)
        A = np.zeros((nNodes, nLabels))

        for i in range(nNodes):
            f = np.where(node_t == i)
            A[i, :] = v[f[1]].T

        cost_matrix = np.max(A) - A

        m = Munkres()
        indexes = m.compute(cost_matrix)
        indexes_ = np.argmin(cost_matrix, axis=-1)

        if len(buffer_ids1) > len(buffer_ids2):
            for i in range(len(buffer_ids2)):
                _, idx1 = indexes[i]
                min_ids[buffer_ids2[i]] = buffer_ids1[idx1]
        else:
            for i in range(len(buffer_ids2)):
                _, idx1 = indexes[i]
                if idx1 >= len(buffer_ids1):
                    continue
                else:
                    min_ids[buffer_ids2[i]] = buffer_ids1[idx1]
        return min_ids

    @staticmethod
    def cvx_opt_matching(M, node_r, node_t, buffer_ids1, buffer_ids2, min_ids):
        """
        Find solution by using cvxopt lib
        """
        M = 1 - M
        N = M.shape[0]
        sqrt_N = int(np.sqrt(N))

        # Find the smallest eigenvalue
        eig_value, _ = np.linalg.eig(M)
        min_eig = int(min(eig_value))

        # Define parameter for QP solver
        # Convex function
        P = np.array(M - min_eig * np.identity(N))
        P = 2 * matrix(P.tolist())
        q = matrix((np.zeros((1, N)) + min_eig).tolist())

        # Constraint
        # x > 0
        G = np.diag(np.diag(- np.ones((N, N))))
        G = matrix(G.tolist())
        h = matrix(0.0, (N, 1))

        # Generate C matrix to enforce x
        C = np.zeros((2 * sqrt_N, N))
        for i in range(sqrt_N):
            C[i, i * sqrt_N: (i + 1) * sqrt_N] = 1

        for i in range(sqrt_N, 2 * sqrt_N):
            row = i - sqrt_N
            for j in range(sqrt_N):
                C[i, row + j * sqrt_N] = 1

        A = matrix(C.T.tolist())
        b = matrix((np.ones((2 * sqrt_N, 1))).T.tolist())

        sol = np.array(solvers.qp(P, q, G, h, A, b, kktsolver='ldl', options={'kktreg': 1e-9})['x'])
        output = np.zeros((sqrt_N, sqrt_N))
        for i in range(sqrt_N):
            output[i, :] = sol[i * sqrt_N: (i + 1) * sqrt_N, 0]
        cost_matrix = np.max(output) - output

        m = Munkres()
        indexes = m.compute(cost_matrix)

        if len(buffer_ids1) > len(buffer_ids2):
            for i in range(len(buffer_ids2)):
                _, idx1 = indexes[i]
                min_ids[buffer_ids2[i]] = buffer_ids1[idx1]
        else:
            for i in range(len(buffer_ids2)):
                _, idx1 = indexes[i]
                if idx1 >= len(buffer_ids1):
                    continue
                else:
                    min_ids[buffer_ids2[i]] = buffer_ids1[idx1]

        return min_ids
