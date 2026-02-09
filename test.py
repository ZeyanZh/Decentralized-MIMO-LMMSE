from sinr_analysis import *



config = Config()
config.M = 15 ##  ## number of users is M + 1 = 31
config.N = 40 ## number of antennas at the BS is 30
config.Ns = [20, 20] ## 3 clusters, each cluster 20 antennas
config.K = len(config.Ns) ## K = 2 clusters
config.sigma2 = 0.001 ## AWGN power \sigma^2 = 0.001
config.wsigma2 = [0.001, 0.005] ##  the resulting noise power \widetilde{\sigma}_k^2
config.set_rho()
config.rhos *= 0.5
sinr = SINRAna(config) # set up the fucntions

N_MC_times = 2000 # number of trails of Monte Carlo
alpha = [1, 1] # the fusion coefficients for LFCC

########### The following is a example for calculating the SINR ########
####################### LFOC #######################
print('SINR with LFOC (analysis):', sinr.get_DE_sinr_opt())
print('SINR with LFOC (simulation):', sinr.get_MC_SINR_opt(T=N_MC_times))
print('\n')
####################### LFSC #######################
print('SINR with LFSC (analysis):', sinr.get_DE_sinr_n_opt())
print('SINR with LFSC (simulation):', sinr.get_MC_SINR_n_opt(T=N_MC_times))
print('\n')

####################### LFCC #######################
print('SINR with LFOC (analysis):', sinr.get_DE_sinr_constant(alpha=alpha))
print('SINR with LFOC (simulation):', sinr.get_MC_SINR_constant(alpha=alpha, T=N_MC_times))
print('\n')

####################################################################
####################### plot the curve SINR vs SNR (LFSC) ###########
import matplotlib.pyplot as plt
from tqdm import tqdm
sinr_Ana_list = []
sinr_Sim_list = []

snr_dB_list = np.linspace(-20, 30, 6)

#### from dB to real value ###
snr_list = [10 ** (x / 10) for x in snr_dB_list]

for snr in tqdm(snr_list):
    sigma2 = 1 / snr ## snr = 1 / \sigma^2
    config.sigma2 = sigma2
    sinr = SINRAna(config)
    sinr_Ana = sinr.get_DE_sinr_n_opt()
    sinr_Sim = sinr.get_MC_SINR_n_opt(T=500) ## 500 trails
    sinr_Ana_list.append(sinr_Ana)
    sinr_Sim_list.append(sinr_Sim)

plt.semilogy(snr_dB_list, sinr_Ana_list, label='Ana.')
plt.semilogy(snr_dB_list,  sinr_Sim_list, ls='', marker='o', label='Sim.')
plt.xlabel('SNR (dB)')
plt.ylabel('SINR')
plt.grid(True, ls='--')
plt.show()

