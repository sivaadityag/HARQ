# -*- coding: utf-8 -*-
# Python 3.8
import numpy as np
from Fasura import FASURA
import time

np.random.seed(0)
# =========================================== Initialization =========================================== #
# Number of active users
K = 16

# number of pilots (length of the pilots)
nPilots = 400

# Spreading sequence length
L = 9

# Length of the code
nc = 512

# Length of the message, Length of the First and Second part of the message
B = 100
Bf = 12
Bs = B - Bf

# Number of spreading sequence/ pilot sequence
J = 2 ** Bf

# Length of List (Decoder)
nL = 64

nChanlUses = int(nPilots + L * nc / 2)
print("Number of channel uses::: " + str(nChanlUses))

# Number of Antennas
M = 4
# EbN0dB
EbN0dB = 1.25

# --- Variance of Noise
sigma2 = 1.0 / ((10 ** (EbN0dB / 10.0)) * B)

# Number of iterations
nIter = 300

# =========================================== Simulation =========================================== #
start = time.time()
print('Number of users: ' + str(K))
print('Number of antennas: ' + str(M))


S = (1 - 2 * np.round(np.random.randint(low=0, high=2, size=(nChanlUses, J)))) + 1j * (1 - 2 * np.round(np.random.randint(low=0, high=2, size=(nChanlUses, J))))

P = S[0:nPilots, :] / np.sqrt(2.0 * nChanlUses)

A = S[nPilots::, :] / np.sqrt(4.0 * nChanlUses)


# --- Generate the channel coefficients
H = (1 / np.sqrt(2)) * (np.random.normal(0, 1, (K, M)) + 1j * np.random.normal(0, 1, (K, M)))


rounds = 9

resultsDE = np.zeros((nIter,rounds))
resultsFA = np.zeros((nIter,rounds))


# --- Generate K msgs at once
msgs = np.random.randint(0, 2, (K, B))

# --- Run the simulation nIter times

for Iter in range(nIter):

    print('======= Iteration number of the Monte Carlo Simulation: ' + str(Iter))

    Y = np.zeros((nChanlUses, M), dtype=complex)

    for round in range(rounds):

        # print("HARQ round:", round)

        # Initialize the scheme and generate the entire "coded" message at once.

        if round == 0:
            # --- Create a FASURA object
            scheme = FASURA(K, nPilots, B, Bf, L, nc, nL, M, sigma2, H, P, A, nChanlUses)

            # --- Encode the data just once and use the TX message in subsequent rounds
            global XH
            XH = scheme.transmitter(msgs, H)

            # --- Generate the noise
            N = (1 / np.sqrt(2)) * (np.random.normal(0, np.sqrt(sigma2), (int(nChanlUses/2), M)) + 1j * np.random.normal(0, np.sqrt(sigma2),(int(nChanlUses/2), M)))


            # --- Transmitted Signal

            Y_temp = XH[0:int(nChanlUses/2), :] + N

            # --- Preprocessing

            for i in range(0, int(nChanlUses/2)):
                Y[i, :] = Y_temp[i, :]

            # --- Decode

            DE, FA, Khat = scheme.receiver(Y)

            if Khat == 0:
                resultsFA[Iter, round] = 0
                resultsDE[Iter, round] = 0
            else:
                resultsFA[Iter, round] = FA / Khat
                resultsDE[Iter, round] = DE / K

        elif round == 1:

            # --- Generate the noise
            N = (1 / np.sqrt(2)) * (np.random.normal(0, np.sqrt(sigma2),(int(nChanlUses/16), M)) + 1j * np.random.normal(0, np.sqrt(sigma2),(int(nChanlUses/16),M)))

            # --- Transmitted Signal

            Y_temp = XH[int(nChanlUses/2):int(9*nChanlUses/16), :] + N

            # --- Preprocessing

            for i in range(int(nChanlUses/2),int(9*nChanlUses/16)):
                Y[i, :] = Y_temp[i-int(nChanlUses/2), :]

            # --- Decode

            DE, FA, Khat = scheme.receiver(Y)

            if Khat == 0:
                resultsFA[Iter, round] = 0
                resultsDE[Iter, round] = 0
            else:
                resultsFA[Iter, round] = FA / Khat
                resultsDE[Iter, round] = DE / K

        elif round == 2:

            # --- Generate the noise
            N = (1 / np.sqrt(2)) * (np.random.normal(0, np.sqrt(sigma2),(int(nChanlUses/16), M)) + 1j * np.random.normal(0, np.sqrt(sigma2),(int(nChanlUses/16), M)))

            # --- Transmitted Signal

            Y_temp = XH[int(9 * nChanlUses / 16):int(5 * nChanlUses / 8), :] + N

            # --- Preprocessing

            for i in range(int(9 * nChanlUses / 16), int(5 * nChanlUses / 8)):
                Y[i, :] = Y_temp[i-int(9 * nChanlUses / 16), :]

            # --- Decode

            DE, FA, Khat = scheme.receiver(Y)

            if Khat == 0:
                resultsFA[Iter, round] = 0
                resultsDE[Iter, round] = 0
            else:
                resultsFA[Iter, round] = FA / Khat
                resultsDE[Iter, round] = DE / K

        elif round == 3:

            # --- Generate the noise
            N = (1 / np.sqrt(2)) * (np.random.normal(0, np.sqrt(sigma2),(int(nChanlUses/16), M)) + 1j * np.random.normal(0, np.sqrt(sigma2),(int(nChanlUses/16), M)))

            # --- Transmitted Signal

            Y_temp = XH[int(5 * nChanlUses / 8):int(11 * nChanlUses / 16), :] + N

            # --- Preprocessing

            for i in range(int(5 * nChanlUses / 8), int(11 * nChanlUses / 16)):
                Y[i, :] = Y_temp[i-int(5 * nChanlUses / 8), :]

            # --- Decode

            DE, FA, Khat = scheme.receiver(Y)

            if Khat == 0:
                resultsFA[Iter, round] = 0
                resultsDE[Iter, round] = 0
            else:
                resultsFA[Iter, round] = FA / Khat
                resultsDE[Iter, round] = DE / K

        elif round == 4:

            # --- Generate the noise
            N = (1 / np.sqrt(2)) * (np.random.normal(0, np.sqrt(sigma2),(int(nChanlUses/16), M)) + 1j * np.random.normal(0, np.sqrt(sigma2),(int(nChanlUses/16), M)))

            # --- Transmitted Signal

            Y_temp = XH[int(11 * nChanlUses / 16):int(3 * nChanlUses / 4), :] + N

            # --- Preprocessing

            for i in range(int(11 * nChanlUses / 16), int(3 * nChanlUses / 4)):
                Y[i, :] = Y_temp[i - int(11 * nChanlUses / 16), :]

            # --- Decode

            DE, FA, Khat = scheme.receiver(Y)

            if Khat == 0:
                resultsFA[Iter, round] = 0
                resultsDE[Iter, round] = 0
            else:
                resultsFA[Iter, round] = FA / Khat
                resultsDE[Iter, round] = DE / K

        elif round == 5:

            # --- Generate the noise
            N = (1 / np.sqrt(2)) * (np.random.normal(0, np.sqrt(sigma2),(int(nChanlUses/16), M)) + 1j * np.random.normal(0, np.sqrt(sigma2),(int(nChanlUses/16), M)))

            # --- Transmitted Signal

            Y_temp = XH[int(3 * nChanlUses / 4):int(13 * nChanlUses / 16), :] + N

            # --- Preprocessing

            for i in range(int(3 * nChanlUses / 4), int(13 * nChanlUses / 16)):
                Y[i, :] = Y_temp[i - int(3 * nChanlUses / 4), :]

            # --- Decode

            DE, FA, Khat = scheme.receiver(Y)

            if Khat == 0:
                resultsFA[Iter, round] = 0
                resultsDE[Iter, round] = 0
            else:
                resultsFA[Iter, round] = FA / Khat
                resultsDE[Iter, round] = DE / K

        elif round == 6:

            # --- Generate the noise
            N = (1 / np.sqrt(2)) * (np.random.normal(0, np.sqrt(sigma2),(int(nChanlUses/16), M)) + 1j * np.random.normal(0, np.sqrt(sigma2),(int(nChanlUses/16), M)))

            # --- Transmitted Signal

            Y_temp = XH[int(13 * nChanlUses / 16):int(7 * nChanlUses / 8), :] + N

            # --- Preprocessing

            for i in range(int(13 * nChanlUses / 16), int(7 * nChanlUses / 8)):
                Y[i, :] = Y_temp[i - int(13 * nChanlUses / 16), :]

            # --- Decode

            DE, FA, Khat = scheme.receiver(Y)

            if Khat == 0:
                resultsFA[Iter, round] = 0
                resultsDE[Iter, round] = 0
            else:
                resultsFA[Iter, round] = FA / Khat
                resultsDE[Iter, round] = DE / K

        elif round == 7:

            # --- Generate the noise
            N = (1 / np.sqrt(2)) * (np.random.normal(0, np.sqrt(sigma2),(int(nChanlUses/16), M)) + 1j * np.random.normal(0, np.sqrt(sigma2),(int(nChanlUses/16), M)))

            # --- Transmitted Signal

            Y_temp = XH[int(7 * nChanlUses / 8):int(15 * nChanlUses / 16), :] + N

            # --- Preprocessing

            for i in range(int(7 * nChanlUses / 8), int(15 * nChanlUses / 16)):
                Y[i, :] = Y_temp[i - int(7 * nChanlUses / 8), :]

            # --- Decode

            DE, FA, Khat = scheme.receiver(Y)

            if Khat == 0:
                resultsFA[Iter, round] = 0
                resultsDE[Iter, round] = 0
            else:
                resultsFA[Iter, round] = FA / Khat
                resultsDE[Iter, round] = DE / K

        elif round == 8:

            # --- Generate the noise
            N = (1 / np.sqrt(2)) * (np.random.normal(0, np.sqrt(sigma2),(int(nChanlUses/16), M)) + 1j * np.random.normal(0, np.sqrt(sigma2),(int(nChanlUses/16), M)))

            # --- Transmitted Signal

            Y_temp = XH[int(15 * nChanlUses / 16)::, :] + N

            # --- Preprocessing

            for i in range(int(15 * nChanlUses / 16), int(nChanlUses)):
                Y[i, :] = Y_temp[i - int(15 * nChanlUses / 16), :]

            # --- Decode

            DE, FA, Khat = scheme.receiver(Y)

            if Khat == 0:
                resultsFA[Iter, round] = 0
                resultsDE[Iter, round] = 0
            else:
                resultsFA[Iter, round] = FA / Khat
                resultsDE[Iter, round] = DE / K

    np.savetxt('resultsDE.out',resultsDE)
    np.savetxt('resultsFA.out',resultsFA)

missed_det = np.zeros(rounds)
false_alarm = np.zeros(rounds)

for round in range(rounds):
    # print('Missed Detection')
    missed_det[round] = sum(1 - resultsDE[:, round]) / float(nIter)
    # print(missed_det[round])

    # print('False Alarm')
    false_alarm[round] = sum(resultsFA[:, round]) / float(nIter)
    # print(false_alarm[round])

    print("Pe for round:", round , missed_det[round] + false_alarm[round])


print("Final simulation result",missed_det + false_alarm)
