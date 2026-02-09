import numpy as np
import scipy.linalg
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm


def real_2_DB(x):
    return 10 * np.log10(x)


def DB_2_real(x):
    return 10 ** (x / 10)


def get_complex_gaussian_RMT(p, n):
    real_X = np.random.randn(p, n) * np.sqrt(1 / 2)
    image_X = np.random.randn(p, n) * np.sqrt(1 / 2)
    return real_X + 1j * image_X


def get_correlation_matrix_random(N):
    Corr = np.zeros([N, N])
    for i in range(N):
        Corr[i, i] = np.random.uniform(1, 2)
    return Corr


def get_correlation_matrix(N):
    I = np.random.uniform(0, 180)
    J = np.random.uniform(0.5, 2)
    ds = np.random.uniform(0.5, 1)
    Corr = generate_corr_matrix(I, J, ds, N)
    return Corr


def get_correlation_matrix_1(N):
    I = np.random.uniform(0.8, 0.9)
    Corr = np.zeros([N, N], dtype=np.complex128)
    for i in range(N):
        for j in range(N):
            Corr[i, j] = I ** (np.abs(i - j))
    return Corr


def generate_corr_matrix(eta, delta, ds, N):
    supp_phi = np.linspace(-180, 180, 500)
    d_phi = supp_phi[1] - supp_phi[0]
    frac = np.sqrt(2 * np.pi * (delta ** 2))
    Corr = np.zeros([N, N], np.complex128)
    for i in range(N):
        for j in range(N):
            v = np.sum(np.exp(2 * np.pi * 1j * ds * (i - j) * np.sin(np.pi * supp_phi / 180) - (supp_phi - eta) ** 2 / (
                        2 * (delta ** 2)))) * d_phi
            Corr[i, j] = v
    return Corr / frac


def get_correlation_matrix_2(N, K):
    I = np.random.uniform(0, 180)
    J = np.random.uniform(0.1, 0.5)
    ds = np.random.uniform(1, 2)
    Corr = np.zeros([N, N], dtype=np.complex128)
    n = N // K
    for k in range(K):
        for l in range(K):
            Corr[k * n:(k + 1) * n, l * n:(l + 1) * n] = generate_corr_matrix(I, J, ds, n)
    return Corr


def get_correlation_matrix_3(N):
    H = get_complex_gaussian_RMT(N, N) / np.sqrt(2 * N)
    return H @ H.conj().T


class Config:
    def __init__(self) -> None:
        self.N = 12  ## number of antennas at the BS
        self.M = 15  ## number of users (M + 1)
        self.Ns = [4, 4, 4] ## number of antennas in the clusters (in this case, 4, 4, 4)
        self.K = len(self.Ns) ## number of clusters
        self.wsigma2 = 1 ## A list of power of the resulting noise, [\widetilde{\sigma}_1^2, \widetilde{\sigma}_2^2, ..., \widetilde{\sigma}_K^2,].  Ref. Eqs. (9) and (10)
        self.sigma2 = 2 ## power of AWGN at the receiver. Ref. Eq. (1)
        self.Zs = [np.zeros([5, 5]), np.zeros([5, 5]), np.zeros([5, 5])] ## Ref. Eq. (17)
        self.MMSE_Zs = True  ## set Z
        self.MMSE_rho = True ## set \rho
        if self.MMSE_rho:
            self.set_rho()
        else:
            self.rhp = self.sigma2 * np.ones(self.K)  ## Ref. Eq. (16)

    def set(self, N, M, Ns, wsigma2, sigma2, rhos=None):
        self.N = N
        self.M = M
        self.Ns = Ns
        self.wsigma2 = wsigma2
        self.sigma2 = sigma2
        self.K = len(self.Ns)
        if rhos is None:
            self.set_rho()
        else:
            self.rhos = rhos

    def set_rho(self):
        self.rhos = np.array([self.sigma2 / self.Ns[k] for k in range(self.K)])

config = Config()

class SINRAna:
    eps = 1e-7
    '''SINR analysis for the DBP system '''
    def __init__(self, config: Config) -> None:
        self.config = config
        self.N = config.N
        self.M = config.M
        self.K = config.K
        self.Ns = config.Ns
        self.sigma2 = config.sigma2
        self.wsigma2 = config.wsigma2  # [wsigma_1, ..., wsigma_K]

        self.N_ls = copy.deepcopy(config.Ns)
        N_ls_tmp = copy.deepcopy(config.Ns)
        N_ls_tmp.insert(0, 0)
        self.N_ls_csum = np.array(N_ls_tmp).cumsum()

        self.I_N = np.eye(self.N)
        self.init_correlations()
        self.init_Mats()
        if self.config.MMSE_Zs:
            self.get_MMSE_Zs()
        else:
            self.Zs = self.config.Zs

        self.init_DEs()

    def init_correlations(self):
        '''Initialization of the correlation matrices. Ref. Eq. (4). self.R is for user 0'''
        self.Rs = []
        self.R = generate_corr_matrix(60, 10, 1, self.N)
        etas = np.linspace(0, 180, self.M + 1)
        deltas = np.linspace(10, 20, self.M + 1)
        self.R = generate_corr_matrix(eta=etas[0], delta=deltas[0], ds=1, N=self.N)
        for i in range(self.M):
            self.Rs.append(generate_corr_matrix(eta=etas[i + 1], delta=deltas[i + 1], ds=1, N=self.N))

    def get_MMSE_Zs(self):
        self.Zs = []
        for k in range(self.K):
            Z = self.get_sub_matrix(self.DK_all, k, k) / self.N_ls[k]
            self.Zs.append(Z)

    def init_Mats(self):
        '''Initialization of the matrices that are listed in the paper. The variables that are without index (for example, self.T, self.Phi) correspond to the correlation related to user 0 (i.e., T_0, \Phi_0)'''
        self.Phis = []
        self.Ts = []
        self.Vs = []
        self.Ks = []
        self.DKs = []
        self.Phi_sqs = []
        self.Ts_sq = []
        self.VPhi_sqs = []
        self.K_all = 0
        self.DK_all = 0
        self.D_wsigma = np.eye(self.N) * 1.0
        for k in range(self.K):
            self.D_wsigma[self.N_ls_csum[k]: self.N_ls_csum[k + 1], self.N_ls_csum[k]:self.N_ls_csum[k + 1]] = \
            self.wsigma2[k] * np.eye(self.N_ls[k])
        for j in range(self.M):
            T_j = self.Rs[j] @ np.linalg.inv(self.D_wsigma + self.Rs[j])
            DR_j = self.get_bdiag_matrix(self.Rs[j])
            DT_j = DR_j @ np.linalg.inv(self.D_wsigma + DR_j)
            Phi_j = DT_j @ (self.D_wsigma + self.Rs[j]) @ DT_j
            K_j = T_j @ self.D_wsigma
            DK_j = DT_j @ self.D_wsigma
            V_j = T_j @ np.linalg.inv(DT_j)
            self.Phis.append(Phi_j)
            self.Ks.append(K_j)
            self.DKs.append(DK_j)
            self.Vs.append(V_j)
            Phi_sq_j = scipy.linalg.sqrtm(Phi_j)
            self.Phi_sqs.append(Phi_sq_j)
            self.VPhi_sqs.append(V_j @ Phi_sq_j)
            self.K_all += K_j
            self.DK_all += DK_j
        self.T = self.R @ np.linalg.inv(self.D_wsigma + self.R)
        self.DR = self.get_bdiag_matrix(self.R)
        self.DT = self.DR @ np.linalg.inv(self.D_wsigma + self.DR)
        self.Phi = self.DT @ (self.D_wsigma + self.R) @ self.DT
        self.Phi_sq = scipy.linalg.sqrtm(self.Phi)
        self.K0 = self.T @ self.D_wsigma
        self.DK0 = self.DT @ self.D_wsigma
        self.V = self.T @ np.linalg.inv(self.DT)
        self.VPhi_sq = self.V @ self.Phi_sq
        self.K_all += self.K0
        self.DK_all += self.DK0

    def get_sub_matrix(self, Mat, k, l):
        return Mat[self.N_ls_csum[k]: self.N_ls_csum[k + 1], self.N_ls_csum[l]:self.N_ls_csum[l + 1]]

    def get_bdiag_matrix(self, Mat):
        M = np.zeros([self.N, self.N], dtype=np.complex128)
        for k in range(self.K):
            M[self.N_ls_csum[k]: self.N_ls_csum[k + 1], self.N_ls_csum[k]:self.N_ls_csum[k + 1]] = self.get_sub_matrix(
                Mat, k, k)
        return M

    def calculate_v(self):
        self.v = np.zeros([self.K, 1], dtype=np.complex128)
        for k in range(self.K):
            self.v[k, 0] = np.trace(self.de_all_cal.Thetas[k] @ self.get_sub_matrix(self.Phi, k, k)) / self.N_ls[k]

    def calculate_J(self):
        self.J = np.diag(1 / (1 + self.v[:, 0]))

    def calculate_Delta(self):
        self.Delta = np.zeros([self.K, self.K], dtype=np.complex128)
        C = self.K_all + self.sigma2 * np.eye(self.N)
        for k in range(self.K):
            for l in range(self.K):
                Phi_lk = self.get_sub_matrix(self.Phi, l, k)
                # KI_kl = self.get_sub_matrix(self.K_all, k, l)
                # if k == l:
                #     KI_kl += self.sigma2 * np.eye(self.N_ls[k])
                item1 = self.de_all_cal.get_Pi_kl_B(k, l, AA=Phi_lk) / np.sqrt(self.N_ls[k] * self.N_ls[l])
                item2 = self.de_all_cal.get_Upsilon_kl(k, l, AA=Phi_lk, BB=self.get_sub_matrix(C, k, l)) / (
                            self.N_ls[k] * self.N_ls[l])
                self.Delta[k, l] = item1 + item2

    def calculate_Delta_I(self):
        self.Delta_I = np.zeros([self.K, self.K], dtype=np.complex128)
        for k in range(self.K):
            for l in range(self.K):
                Phi_lk = self.get_sub_matrix(self.Phi, l, k)
                Upsilon = 0
                if k == l:
                    KI_kl = self.get_sub_matrix(self.DK_all, k, l)
                    KI_kl += self.sigma2 * np.eye(self.N_ls[k])
                    Upsilon = self.de_all_cal.get_Upsilon_kl(k, l, AA=Phi_lk, BB=KI_kl) / (self.N_ls[k] * self.N_ls[l])
                Pi = self.de_all_cal.get_Pi_kl(k, l, AA=Phi_lk) / np.sqrt(self.N_ls[k] * self.N_ls[l])
                self.Delta_I[k, l] = Upsilon + Pi

    def init_DEs(self):
        self.de_all_cal = DEAllCal(N=self.N,
                                   M=self.M,
                                   K=self.K,
                                   Ns=self.Ns,
                                   sigma2=self.sigma2,
                                   wsigma2=self.wsigma2,
                                   zs=-np.array(self.config.rhos),
                                   Zs=self.Zs,
                                   As=self.Phi_sqs,
                                   Bs=self.VPhi_sqs)
        # print(self.K)
        self.calculate_v()
        self.calculate_J()
        self.calculate_Delta()
        self.calculate_Delta_I()

    def reset_antenna_cluster(self, Ns):
        self.Ns = copy.deepcopy(Ns)
        self.N_ls = copy.deepcopy(Ns)
        N_ls_tmp = copy.deepcopy(Ns)
        N_ls_tmp.insert(0, 0)
        self.N_ls_csum = np.array(N_ls_tmp).cumsum()
        self.K = len(Ns)
        self.init_Mats()
        self.get_MMSE_Zs()
        self.init_DEs()

    def get_H_h_realization(self):
        CG_Mat = get_complex_gaussian_RMT(self.N, self.M)
        CG_Vec = get_complex_gaussian_RMT(self.N, 1)
        Hat_H = self.get_Hat_H(CG_Mat)
        Hat_h = self.get_Hat_h(CG_Vec)
        Tilde_H = self.get_Tilde_H(CG_Mat)
        Tilde_h = self.get_Tilde_h(CG_Vec)
        return Hat_H, Hat_h, Tilde_H, Tilde_h

    def get_Hat_H(self, CG_Mat):
        H = np.zeros([self.N, self.M], dtype=np.complex128)
        for i in range(self.M):
            H[:, i:i + 1] = self.Phi_sqs[i] @ CG_Mat[:, i:i + 1]
        return H

    def get_Hat_h(self, CG_Vec):
        return self.Phi_sq @ CG_Vec

    def get_Tilde_H(self, CG_Mat):
        H = np.zeros([self.N, self.M], dtype=np.complex128)
        for i in range(self.M):
            H[:, i:i + 1] = self.VPhi_sqs[i] @ CG_Mat[:, i:i + 1]
        return H

    def get_Qs(self, Hat_H):
        Hat_Hs = [Hat_H[self.N_ls_csum[k]: self.N_ls_csum[k + 1], :] for k in range(self.K)]
        Qs = []
        for k in range(self.K):
            Qk = np.linalg.inv(
                Hat_Hs[k] @ Hat_Hs[k].conj().T / self.N_ls[k] + self.Zs[k] + self.config.rhos[k] * np.eye(self.N_ls[k]))
            Qs.append(Qk)
        return Qs

    def get_Tilde_h(self, CG_Vec):
        return self.VPhi_sq @ CG_Vec

    ################## Monte-Carlo (MC) SINR code ###############
    ###### The map of function names -> futions schemes : opt -> LFOC, n_opt -> LFSC, constant -> LFCC
    ###### The function names with '_' at the end represent one trail for MC 
    ###### The function names with 'list' at the end represent the whole MC list without average
    
    def get_MC_SINR_opt_(self):
        Hat_H, Hat_h, Tilde_H, Tilde_h = self.get_H_h_realization()
        Hat_Hs = [Hat_H[self.N_ls_csum[k]: self.N_ls_csum[k + 1], :] for k in range(self.K)]
        Hat_hs = [Hat_h[self.N_ls_csum[k]: self.N_ls_csum[k + 1], :] for k in range(self.K)]
        Tilde_Hs = [Tilde_H[self.N_ls_csum[k]: self.N_ls_csum[k + 1], :] for k in range(self.K)]
        Tilde_hs = [Tilde_h[self.N_ls_csum[k]: self.N_ls_csum[k + 1], :] for k in range(self.K)]
        C = self.K_all + self.sigma2 * np.eye(self.N)
        Qs = self.get_Qs(Hat_H)

        Wg_v = np.zeros([self.K, 1], dtype=np.complex128)
        for k in range(self.K):
            Wg_v[k, 0] = (Hat_hs[k].conj().T @ Qs[k] @ Tilde_hs[k])[0, 0] / self.N_ls[k]
        # print(Wg_v)

        G_Delta = np.zeros([self.K, self.K], dtype=np.complex128)
        for k in range(self.K):
            for l in range(self.K):
                G_Delta[k, l] = Hat_hs[k].conj().T @ Qs[k] @ (
                            Tilde_Hs[k] @ Tilde_Hs[l].conj().T + self.get_sub_matrix(C, k, l)) @ Qs[l] @ Hat_hs[l] / (
                                            self.N_ls[k] * self.N_ls[l])
        return (Wg_v.conj().T @ np.linalg.inv(G_Delta) @ Wg_v).real

    def get_MC_SINR_opt(self, T=100):
        ''' T: number of trials for Monte-Carlo'''
        sinr_opt = 0
        for t in range(T):
            sinr_opt += self.get_MC_SINR_opt_()
        return sinr_opt[0, 0] / T

    def get_MC_SINR_opt_list(self, T=100):
        sinr_opt_ls = []
        for t in range(T):
            sinr_opt_ls.append(self.get_MC_SINR_opt_()[0, 0])
        return sinr_opt_ls

    def get_MC_SINR_n_opt_(self):
        Hat_H, Hat_h, Tilde_H, Tilde_h = self.get_H_h_realization()
        Hat_Hs = [Hat_H[self.N_ls_csum[k]: self.N_ls_csum[k + 1], :] for k in range(self.K)]
        Hat_hs = [Hat_h[self.N_ls_csum[k]: self.N_ls_csum[k + 1], :] for k in range(self.K)]
        Tilde_Hs = [Tilde_H[self.N_ls_csum[k]: self.N_ls_csum[k + 1], :] for k in range(self.K)]
        Tilde_hs = [Tilde_h[self.N_ls_csum[k]: self.N_ls_csum[k + 1], :] for k in range(self.K)]
        CDK = self.DK_all + self.sigma2 * np.eye(self.N)
        CK = self.K_all + self.sigma2 * np.eye(self.N)
        Qs = self.get_Qs(Hat_H)

        Wg_v = np.zeros([self.K, 1], dtype=np.complex128)
        g_v = np.zeros([self.K, 1], dtype=np.complex128)
        for k in range(self.K):
            g_v[k, 0] = (Hat_hs[k].conj().T @ Qs[k] @ Hat_hs[k]) / self.N_ls[k]
            Wg_v[k, 0] = (Hat_hs[k].conj().T @ Qs[k] @ Tilde_hs[k]) / self.N_ls[k]

        G_Delta_I = np.zeros([self.K, self.K], dtype=np.complex128)
        G_Delta = np.zeros([self.K, self.K], dtype=np.complex128)
        for k in range(self.K):
            for l in range(self.K):
                G_Delta[k, l] = Hat_hs[k].conj().T @ Qs[k] @ (
                            Tilde_Hs[k] @ Tilde_Hs[l].conj().T + self.get_sub_matrix(CK, k, l)) @ Qs[l] @ Hat_hs[l] / (
                                            self.N_ls[k] * self.N_ls[l])
                G_Delta_I[k, l] = Hat_hs[k].conj().T @ Qs[k] @ (
                            Hat_Hs[k] @ Hat_Hs[l].conj().T + self.get_sub_matrix(CDK, k, l)) @ Qs[l] @ Hat_hs[l] / (
                                              self.N_ls[k] * self.N_ls[l])

        G_Delta_inv = np.linalg.inv(G_Delta)
        return np.abs(g_v.conj().T @ G_Delta_inv @ Wg_v) ** 2 / (
                    g_v.conj().T @ G_Delta_inv @ G_Delta @ G_Delta_inv @ g_v).real

    def get_MC_SINR_n_opt(self, T=100):
        sinr_n_opt = 0
        for t in range(T):
            sinr_n_opt += self.get_MC_SINR_n_opt_()

        return sinr_n_opt[0, 0] / T

    def get_MC_SINR_n_opt_list(self, T=100):
        sinr_n_opt_ls = []
        for t in range(T):
            sinr_n_opt_ls.append(self.get_MC_SINR_n_opt_()[0, 0])

        return sinr_n_opt_ls

    def get_MC_SINR_constant_(self, alpha):
        alpha = np.array(alpha)[None, :]
        Hat_H, Hat_h, Tilde_H, Tilde_h = self.get_H_h_realization()
        Hat_Hs = [Hat_H[self.N_ls_csum[k]: self.N_ls_csum[k + 1], :] for k in range(self.K)]
        Hat_hs = [Hat_h[self.N_ls_csum[k]: self.N_ls_csum[k + 1], :] for k in range(self.K)]
        Tilde_Hs = [Tilde_H[self.N_ls_csum[k]: self.N_ls_csum[k + 1], :] for k in range(self.K)]
        Tilde_hs = [Tilde_h[self.N_ls_csum[k]: self.N_ls_csum[k + 1], :] for k in range(self.K)]
        CDK = self.DK_all + self.sigma2 * np.eye(self.N)
        CK = self.K_all + self.sigma2 * np.eye(self.N)
        Qs = self.get_Qs(Hat_H)

        FF = np.zeros([self.K], dtype=np.complex128)
        Wg_vF = np.zeros([self.K, 1], dtype=np.complex128)
        for k in range(self.K):
            FF[k] = 1 / (1 + Hat_hs[k].conj().T @ Qs[k] @ Hat_hs[k] / self.N_ls[k])
            Wg_vF[k, 0] = FF[k] * (Hat_hs[k].conj().T @ Qs[k] @ Tilde_hs[k]) / (self.N_ls[k])

        item_1 = np.abs(alpha @ Wg_vF)[0, 0].real ** 2

        G_Delta_F = np.zeros([self.K, self.K], dtype=np.complex128)
        for k in range(self.K):
            for l in range(self.K):
                G_Delta_F[k, l] = FF[k] * FF[l] * Hat_hs[k].conj().T @ Qs[k] @ (
                            Tilde_Hs[k] @ Tilde_Hs[l].conj().T + self.get_sub_matrix(CK, k, l)) @ Qs[l] @ Hat_hs[l] / (
                                              self.N_ls[k] * self.N_ls[l])

        item_2 = (alpha @ G_Delta_F @ alpha.conj().T)[0, 0].real

        return item_1 / item_2

    def get_MC_SINR_constant(self, alpha, T=100):
        sinr_cons = 0
        for t in range(T):
            sinr_cons += self.get_MC_SINR_constant_(alpha)

        return sinr_cons / T

    def get_MC_SINR_constant_list(self, alpha, T=100):
        sinr_cons_ls = []
        for t in range(T):
            sinr_cons_ls.append(self.get_MC_SINR_constant_(alpha))

        return sinr_cons_ls

    ################## Deterministic Equivalent (DE) SINR code ##################
    ############################################################################

    def get_DE_sinr_opt(self):
        ''' SINR with LFOC '''
        return (self.v.conj().T @ np.linalg.inv(self.Delta) @ self.v)[0, 0].real

    def get_DE_sinr_n_opt(self):
        ''' SINR with LFSC '''
        Delta_I_inv = np.linalg.inv(self.Delta_I)
        return ((self.v.conj().T @ Delta_I_inv @ self.v).real ** 2 / (
                    self.v.conj().T @ Delta_I_inv @ self.Delta @ Delta_I_inv @ self.v))[0, 0].real

    def get_DE_sinr_constant(self, alpha):
        ''' SINR with LFCC '''
        alpha = np.array(alpha)[None, :]
        item1 = np.abs(alpha @ self.J @ self.v) ** 2
        item2 = np.abs(alpha @ self.J @ self.Delta @ self.J @ alpha.conj().T)
        return item1[0, 0] / item2[0, 0].real

class DECal:
    eps = 1e-7
    ''' The fixed-point algorithm for computing the deterministic equivalents for the resolvent. Ref Eqs. (57) and (58)'''
    def __init__(self, N, M, S, corrs) -> None:
        self.N = N
        self.M = M
        self.S = S
        self.Rs = corrs
        self.I_N = np.eye(self.N, dtype=np.complex128)

    def calculate_DE(self, z):
        ''' The fixed-point algorithm'''
        T = 100000
        self.Theta = np.zeros([self.N, self.N], dtype=np.complex128)
        self.delta_old = np.ones(self.M, dtype=np.complex128)
        self.delta = np.ones(self.M, dtype=np.complex128)
        for t in range(T):
            Omega = 0
            for i in range(self.M):
                Omega += self.Rs[i] / (1 + self.delta_old[i])
            self.Theta = np.linalg.inv(-z * self.I_N + self.S + 1 / self.N * Omega)
            self.delta = np.array([1 / self.N * np.trace(self.Rs[i] @ self.Theta) for i in range(self.M)])
            if np.max(np.abs(self.delta - self.delta_old)) < DECal.eps:
                break
            self.delta_old = self.delta.copy()

class DEAllCal:
    ''' Calculate the deterministic equivalents listed in Lemma 3 and Table III'''
    def __init__(self, N, M, K, Ns, sigma2, wsigma2, zs, Zs, As, Bs) -> None:
        self.N = N
        self.M = M
        self.K = K
        self.Zs = Zs
        self.sigma2 = sigma2
        self.wsigma2 = wsigma2
        self.zs = zs
        self.N_ls = copy.deepcopy(Ns)
        N_ls_tmp = copy.deepcopy(Ns)
        N_ls_tmp.insert(0, 0)
        self.N_ls_csum = np.array(N_ls_tmp).cumsum()

        self.I_N = np.eye(self.N)
        self.I_M = np.eye(self.M)
        self.As = As
        self.Bs = Bs
        self.Omegas = [As[i] @ As[i].conj().T for i in range(self.M)]
        self.BBs = [Bs[i] @ Bs[i].conj().T for i in range(self.M)]
        self.BAs = [Bs[i] @ As[i].conj().T for i in range(self.M)]
        self.ABs = [As[i] @ Bs[i].conj().T for i in range(self.M)]
        self.calculate_DE()

    def get_sub_matrix(self, Mat, k, l):
        return Mat[self.N_ls_csum[k]: self.N_ls_csum[k + 1], self.N_ls_csum[l]:self.N_ls_csum[l + 1]]

    def get_deltas_Thetas(self):
        self.deltas = []
        self.Thetas = []
        for k in range(self.K):
            Cs = [self.get_sub_matrix(self.Omegas[i], k, k) for i in range(self.M)]
            de = DECal(self.N_ls[k], self.M, self.Zs[k], Cs)
            de.calculate_DE(self.zs[k])
            self.deltas.append(de.delta)
            self.Thetas.append(de.Theta)

    def get_Gamma_kl(self, k, l):
        Gamma_kl = np.zeros([self.M, self.M], dtype=np.complex128)
        for i in range(self.M):
            for j in range(self.M):
                Gamma_kl[i, j] = np.trace(
                    self.get_sub_matrix(self.Omegas[i], l, k) @ self.Thetas[k] @ self.get_sub_matrix(self.Omegas[j], k,
                                                                                                     l) @ self.Thetas[
                        l])
        return Gamma_kl / (self.N_ls[k] * self.N_ls[l])

    def get_Gammas(self):
        self.Gammas = [[] for i in range(self.K)]
        for k in range(self.K):
            for l in range(self.K):
                Gamma_kl = self.get_Gamma_kl(k, l)
                self.Gammas[k].append(Gamma_kl)

    def get_WFs(self):
        self.WFs = []
        for k in range(self.K):
            self.WFs.append(np.diag(1 / (1 + self.deltas[k])))

    def get_Xis(self):
        self.Xis = [[] for i in range(self.K)]
        for k in range(self.K):
            for l in range(self.K):
                D_F_l = self.WFs[l]
                D_F_k = self.WFs[k]
                self.Xis[k].append(np.linalg.inv(self.I_M - self.Gammas[k][l] @ D_F_k @ D_F_l))

    def get_W_Xis(self):
        self.W_Xis = [[] for i in range(self.K)]
        for k in range(self.K):
            for l in range(self.K):
                D_F_l = self.WFs[l]
                D_F_k = self.WFs[k]
                self.W_Xis[k].append(np.linalg.inv(self.I_M - D_F_k @ D_F_l @ self.Gammas[k][l]))

    def calculate_DE(self):
        self.get_deltas_Thetas()
        self.get_WFs()
        self.get_Gammas()
        self.get_Xis()
        self.get_W_Xis()

    def get_Wlambda_kl(self, k, l, AA):
        Wlamda = np.zeros([1, self.M], dtype=np.complex128)
        for j in range(self.M):
            Wlamda[0, j] = np.trace(
                AA @ self.Thetas[k] @ self.get_sub_matrix(self.Omegas[j], k, l) @ self.Thetas[l]) / (
                               np.sqrt(self.N_ls[k] * self.N_ls[l]))
        return Wlamda

    def get_lambda_kl(self, k, l, BB):
        lamda = np.zeros([self.M, 1], dtype=np.complex128)
        for j in range(self.M):
            lamda[j, 0] = np.trace(self.get_sub_matrix(self.Omegas[j], l, k) @ self.Thetas[k] @ BB @ self.Thetas[l]) / (
                np.sqrt(self.N_ls[k] * self.N_ls[l]))
        return lamda

    def get_Wlambda_kl_VW(self, k, l, VW, AA):
        Wlamda_VW = np.zeros([1, self.M], dtype=np.complex128)
        for j in range(self.M):
            Wlamda_VW[0, j] = np.trace(AA @ self.Thetas[k] @ self.get_sub_matrix(VW[j], k, l) @ self.Thetas[l]) / (
                np.sqrt(self.N_ls[k] * self.N_ls[l]))
        return Wlamda_VW

    def get_Lambda_kl_VW(self, k, l, VW):
        Lamda_kl_VW = np.zeros([self.M, self.M], dtype=np.complex128)
        for i in range(self.M):
            for j in range(self.M):
                Lamda_kl_VW[i, j] = np.trace(
                    self.get_sub_matrix(self.Omegas[i], l, k) @ self.Thetas[k] @ self.get_sub_matrix(VW[j], k, l) @
                    self.Thetas[l])
        return Lamda_kl_VW / (self.N_ls[k] * self.N_ls[l])

    def get_D_k_VW(self, k, VW):
        D_k_VW = np.zeros([self.M, self.M], dtype=np.complex128)
        for j in range(self.M):
            D_k_VW[j, j] = np.trace(self.get_sub_matrix(VW[j], k, k) @ self.Thetas[k]) / self.N_ls[k]
        return D_k_VW

    def get_Pi_kl_B(self, k, l, AA):
        ones_M = np.ones([self.M, 1])
        Wlamda = self.get_Wlambda_kl(k, l, AA)
        Wlamda_BB = self.get_Wlambda_kl_VW(k, l, self.BBs, AA)
        Wlamda_BA = self.get_Wlambda_kl_VW(k, l, self.BAs, AA)
        Wlamda_AB = self.get_Wlambda_kl_VW(k, l, self.ABs, AA)
        D_lAB = self.get_D_k_VW(l, self.ABs)
        D_kBA = self.get_D_k_VW(k, self.BAs)
        Lamda_BB = self.get_Lambda_kl_VW(k, l, self.BBs)
        Lamda_BA = self.get_Lambda_kl_VW(k, l, self.BAs)
        Lamda_AB = self.get_Lambda_kl_VW(k, l, self.ABs)
        Xi_kl = self.Xis[k][l]
        DF_l = D_lAB @ self.WFs[l]
        DF_k = D_kBA @ self.WFs[k]
        item_1 = (Wlamda_BB - Wlamda_BA @ DF_l - Wlamda_AB @ DF_k) @ ones_M
        item_2 = Wlamda @ self.WFs[k] @ self.WFs[l] @ Xi_kl @ (
                    Lamda_BB - Lamda_BA @ DF_l - Lamda_AB @ DF_k + D_lAB @ D_kBA) @ ones_M
        return item_1[0, 0] + item_2[0, 0]

    def get_Upsilon_kl(self, k, l, AA, BB):
        item1 = np.trace(AA @ self.Thetas[k] @ BB @ self.Thetas[l])
        Wlamda = self.get_Wlambda_kl(k, l, AA)
        lamda = self.get_lambda_kl(k, l, BB)
        item2 = Wlamda @ self.WFs[k] @ self.WFs[l] @ self.Xis[k][l] @ lamda
        return item1 + item2[0, 0]

    def get_Pi_kl(self, k, l, AA):
        ones_M = np.ones([self.M, 1])
        Wlamda = self.get_Wlambda_kl(k, l, AA)
        return (Wlamda @ self.WFs[k] @ self.WFs[l] @ self.Xis[k][l] @ ones_M)[0, 0]

