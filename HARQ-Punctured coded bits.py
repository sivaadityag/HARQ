# -*- coding: utf-8 -*-
# Python 3.8
import numpy as np
from Fasura import FASURA
import time
from utility import bin2dec, dec2bin, crcEncoder, crcDecoder

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
EbN0dB = -1

# --- Variance of Noise
sigma2 = 1.0 / ((10 ** (EbN0dB / 10.0)) * B)

# Number of iterations
nIter = 1

# =========================================== Simulation =========================================== #
start = time.time()
print('Number of users: ' + str(K))
print('Number of antennas: ' + str(M))


S = (1 - 2 * np.round(np.random.randint(low=0, high=2, size=(nChanlUses, J)))) + 1j * (1 - 2 * np.round(np.random.randint(low=0, high=2, size=(nChanlUses, J))))

P = S[0:nPilots, :] / np.sqrt(2.0 * nChanlUses)

A = S[nPilots::, :] / np.sqrt(4.0 * nChanlUses)


# --- Generate the channel coefficients
H = (1 / np.sqrt(2)) * (np.random.normal(0, 1, (K, M)) + 1j * np.random.normal(0, 1, (K, M)))


rounds = 3

resultsDE = np.zeros((nIter,rounds))
resultsFA = np.zeros((nIter,rounds))


# --- Generate K msgs at once
msgs = np.random.randint(0, 2, (K, B))

# --- Run the simulation nIter times

for Iter in range(nIter):

    print('======= Iteration number of the Monte Carlo Simulation: ' + str(Iter))

    for round in range(rounds):

        # print("HARQ round:", round)

        # Initialize the scheme and generate the entire "coded" message at once.

        if round == 0:
            print("In round:::", round + 1)

            # --- Create a FASURA object
            scheme = FASURA(K, nPilots, B, Bf, L, nc, nL, M, sigma2, H, P, A, nChanlUses)

            # --- Encode the data just once and use the TX message in subsequent rounds
            global XH

            XH = scheme.transmitter(msgs, H)

            # --- Generate the noise
            N = (1 / np.sqrt(2)) * (np.random.normal(0, np.sqrt(sigma2), (int(nChanlUses), M)) + 1j * np.random.normal(0, np.sqrt(sigma2),(int(nChanlUses), M)))

            # --- Full version of the transmitted Signal

            Y = XH + N

            Y_final = Y.copy()

            # --- Preprocessing

            Y_temp1 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, int((1/2) * nChanlUses)):
                Y_temp1[i, :] = Y_final[i, :]


            # --- Decode

            DE1, FA, Khat, Y_decoded1, XX1 = scheme.receiver(Y_temp1, 0, int((1/2) * nChanlUses), flag=0)
            decoded_indices = np.array([], dtype=int)

            for i in range(XX1.shape[0]):
                decoded_indices = np.append(decoded_indices, int(bin2dec(XX1[i, 0:Bf])))

            print("Round 1 status::", DE1, FA, Khat, XX1.shape, Y_decoded1.shape, decoded_indices)

            # --- Pre processing for next round of Tx

            Y_final = Y_final - Y_decoded1

            if Khat == 0:
                resultsFA[Iter, round] = 0
                resultsDE[Iter, round] = 0
            else:
                resultsFA[Iter, round] = FA / Khat
                resultsDE[Iter, round] = DE1 / K

        elif round == 1:

            print("In round:::", round+1)

            # --- Preprocessing
            Y_temp2 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, int((3/4) * nChanlUses)):
                Y_temp2[i, :] = Y_final[i, :]

            # --- Decode

            DE2, FA, Khat, Y_decoded2, XX2 = scheme.receiver(Y_temp2, DE1, int((3/4) * nChanlUses), flag=1)

            decoded_indices = np.array([], dtype=int)

            for i in range(XX2.shape[0]):
                decoded_indices = np.append(decoded_indices, int(bin2dec(XX2[i, 0:Bf])))

            print("Round 2 status::", DE1+DE2, FA, Khat, XX2.shape, Y_decoded2.shape, decoded_indices)

            # --- Pre processing for next round of Tx

            Y_final = Y_final - Y_decoded2

            if Khat == 0:
                resultsFA[Iter, round] = 0
                resultsDE[Iter, round] = 0
            else:
                resultsFA[Iter, round] = FA / Khat
                resultsDE[Iter, round] = DE2 / K

        elif round == 2:

            print("In round:::", round+1)

            # --- Preprocessing
            Y_temp3 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, int(nChanlUses)):
                Y_temp3[i, :] = Y_final[i, :]

            # --- Decode

            DE3, FA, Khat, Y_decoded3, XX3 = scheme.receiver(Y_temp3, DE1+DE2, int(nChanlUses), flag=1)

            decoded_indices = np.array([], dtype=int)

            for i in range(XX3.shape[0]):
                decoded_indices = np.append(decoded_indices, int(bin2dec(XX3[i, 0:Bf])))

            print("Round 3 status::", DE1 + DE2 + DE3, FA, Khat, XX3.shape, Y_decoded3.shape, decoded_indices)

            # --- Pre processing for next round of Tx

            Y_final = Y_final - Y_decoded3

            if Khat == 0:
                resultsFA[Iter, round] = 0
                resultsDE[Iter, round] = 0
            else:
                resultsFA[Iter, round] = FA / Khat
                resultsDE[Iter, round] = DE3 / K

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
