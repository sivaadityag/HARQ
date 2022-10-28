# -*- coding: utf-8 -*-
# Python 3.8
import numpy as np
from Try.Fasura import FASURA
import time

np.random.seed(0)
# =========================================== Initialization =========================================== #
# Number of active users
K = 16

# number of pilots (length of the pilots)
nPilots = np.array([128, 344, 768, 1024]).astype(dtype=int)

# Spreading sequence length
L = np.array([9,9,9,9]).astype(dtype=int)

# Length of the code
nc = 512

# Length of the message, Length of the First and Second part of the message
B = 100
Bf = np.array([10, 12, 14, 16]).astype(dtype=int)
Bs = B - Bf

# Number of spreading sequence/ pilot sequence
J = 2 ** Bf

# Length of List (Decoder)
nL = 64

nChanlUses = np.zeros(len(nPilots)).astype(dtype=int)

missed_det = np.zeros(len(nPilots))
false_alarm = np.zeros(len(nPilots))

# nChanlUses = int((nc / np.log2(4)) * L + nPilots)
# if nChanlUses > 3200:
#    print("The length of the channel input is larger than 3200")
#    sys.exit()


# Number of Antennas
M = 4
# EbN0dB
EbN0dB = np.array([1, 1.25, 2.5, 3.25])

# Number of iterations
nIter = 50

# To save the results
resultsFA = np.zeros(nIter)  # For False Alarm
resultsDE = np.zeros(nIter)  # For Detection

# =========================================== Simulation =========================================== #
start = time.time()
print('Number of users: ' + str(K))
print('Number of antennas: ' + str(M))

# --- Variance of Noise
sigma2 = 1.0 / ((10 ** (EbN0dB / 10.0)) * B)
print('Sigma^2: ' + str(sigma2))
# --- Run the simulation nIter times

for pilot_size in range(len(nPilots)):

    nChanlUses[pilot_size] = int((nc / np.log2(4)) * L[pilot_size] + nPilots[pilot_size])
    print("Number of channel uses::: " + str(nChanlUses[pilot_size]))

    print("Number of pilots in Tx::: " + str(pilot_size) + "\n" + str(nPilots[pilot_size]))

    np.random.seed(0)

    S = (1 - 2 * np.round(np.random.randint(low=0, high=2, size=(nChanlUses[pilot_size], J[pilot_size])))) + 1j * (
            1 - 2 * np.round(np.random.randint(low=0, high=2, size=(nChanlUses[pilot_size], J[pilot_size]))))

    P = S[0:nPilots[pilot_size], :] / np.sqrt(2.0 * nChanlUses[pilot_size])

    A = S[nPilots[pilot_size]::, :] / np.sqrt(4.0 * nChanlUses[pilot_size])

    for Iter in range(nIter):
        # print()
        print('======= Iteration number of the Monte Carlo Simulation: ' + str(Iter))
        # print()

        # --- Generate K msgs
        msgs = np.random.randint(0, 2, (K, B))

        # --- Generate the channel coefficients
        H = (1 / np.sqrt(2)) * (np.random.normal(0, 1, (K, M)) + 1j * np.random.normal(0, 1, (K, M)))

        # --- Create a FASURA object
        scheme = FASURA(K, nPilots[pilot_size], B, Bf[pilot_size], L[pilot_size], nc, nL, M, sigma2[pilot_size], H, P, A,
                        nChanlUses[pilot_size])

        # --- Encode the data
        XH = scheme.transmitter(msgs, H)

        # --- Generate the noise
        N = (1 / np.sqrt(2)) * (
                np.random.normal(0, np.sqrt(sigma2[pilot_size]), (nChanlUses[pilot_size], M)) + 1j * np.random.normal(0, np.sqrt(
            sigma2[pilot_size]), (nChanlUses[pilot_size], M)))

        # --- Received Signal
        Y = XH + N

        # --- Decode
        DE, FA, Khat = scheme.receiver(Y)
        if Khat == 0:
            resultsFA[Iter] = 0
            resultsDE[Iter] = 0
        else:
            resultsFA[Iter] = FA / Khat
            resultsDE[Iter] = DE / K

    print('Missed Detection')
    missed_det[pilot_size] = sum(1 - resultsDE) / float(nIter)
    # print(sum(1 - resultsDE) / float(nIter))
    print(missed_det[pilot_size])

    print('False Alarm')
    false_alarm[pilot_size] = sum(resultsFA) / float(nIter)
    # print(sum(resultsFA) / float(nIter))
    print(false_alarm[pilot_size])
print(missed_det+false_alarm)
    # print('Total time')
    # print(time.time() - start)
    # print()
