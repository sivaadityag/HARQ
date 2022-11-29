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
nPilots = 896

# Spreading sequence length
L = 9

# Length of the code
nc = 512

# Length of the message, Length of the First and Second part of the message
B = 100
Bf = 16
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
EbN0dB = 3

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


rounds = 1

resultsDE = np.zeros((nIter,rounds))
resultsFA = np.zeros((nIter,rounds))


# --- Generate K msgs at once
msgs = np.random.randint(0, 2, (K, B))

# --- HARQ channel uses

nChanlUses_harq = int((L * nc)/4)
nChanlUses_0 = nPilots + nChanlUses_harq

# --- Run the simulation nIter times

for Iter in range(nIter):

    print('======= Iteration number of the Monte Carlo Simulation: ' + str(Iter))

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
            N = (1 / np.sqrt(2)) * (np.random.normal(0, np.sqrt(sigma2), (int(nChanlUses), M)) + 1j * np.random.normal(0, np.sqrt(sigma2),(int(nChanlUses), M)))

            # --- Full version of the transmitted Signal

            Y = XH + N

            Y_final1 = Y.copy()
            Y_final2 = Y.copy()

            print("In round:::", round + 1)

            # --- Preprocessing
            Y_temp1 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, nChanlUses_0):
                Y_temp1[i, :] = Y_final1[i, :]

            # --- Decode

            DE1, FA, Khat, Y_detected1, msgs1, Y_decoded1, HARQ_exit, HARQ_curr = scheme.receiver(Y_temp1, 0, nChanlUses_0, None)

            if HARQ_exit == 1:
                break

            elif HARQ_curr == 1:
                detected_indices = np.array([], dtype=int)

                for i in range(msgs1.shape[0]):
                    detected_indices = np.append(detected_indices, int(bin2dec(msgs1[i, 0:Bf])))

                print("Round 1 status::", DE1, FA, Khat, msgs1.shape, detected_indices)

                # Pre-processing for next round

                Y_final1 = Y_final1 - Y_decoded1
                Y_final2 = Y_final2 - Y_detected1

            else:
                print("Moving onto next round if available")
                msgs1 = np.array([], dtype=int)
            # if Khat == 0:
            #     resultsFA[Iter, round] = 0
            #     resultsDE[Iter, round] = 0
            # else:
            #     resultsFA[Iter, round] = FA / Khat
            #     resultsDE[Iter, round] = DE1 / K

        elif round == 1:

            print("In round:::", round + 1)

            # --- Preprocessing
            Y_temp2 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, nChanlUses_0):
                Y_temp2[i, :] = Y_final1[i, :]

            for i in range(nChanlUses_0, nChanlUses_0 + int((1/4) * nChanlUses_harq)):
                Y_temp2[i, :] = Y_final2[i, :]

            # --- Decode

            # Check if empty detected msgs from previous rounds

            if msgs1.shape[0] == 0:
                old_msgs = None
            else:
                old_msgs = msgs1

            DE2, FA, Khat, Y_detected2, msgs2, Y_decoded2, HARQ_exit, HARQ_curr = scheme.receiver(Y_temp2, DE1, nChanlUses_0 + int((1/4) * nChanlUses_harq) , old_msgs)

            # Pre-processing for next round

            if HARQ_exit == 1:
                break

            elif HARQ_curr == 1:

                detected_indices = np.array([], dtype=int)

                for i in range(msgs2.shape[0]):
                    detected_indices = np.append(detected_indices, int(bin2dec(msgs2[i, 0:Bf])))

                print("Round 2 status::", DE1 + DE2, FA, Khat, msgs2.shape, detected_indices)

                # Pre-processing for next round

                Y_final1 = Y_final1 - Y_decoded2
                Y_final2 = Y_final2 - Y_detected2
            else:
                print("Moving onto next round if available")
                msgs2 = np.array([], dtype=int)

            # if Khat == 0:
            #     resultsFA[Iter, round] = 0
            #     resultsDE[Iter, round] = 0
            # else:
            #     resultsFA[Iter, round] = FA / Khat
            #     resultsDE[Iter, round] = DE2 / K

        elif round == 2:

            print("In round:::", round + 1)

            # --- Preprocessing
            Y_temp3 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, nChanlUses_0 + int((1/4) * nChanlUses_harq)):
                Y_temp3[i, :] = Y_final1[i, :]

            for i in range(nChanlUses_0 + int((1/4) * nChanlUses_harq), nChanlUses_0 + int((2/4) * nChanlUses_harq)):
                Y_temp3[i, :] = Y_final2[i, :]

            # --- Decode

            # Checking the status of previous round detected messages

            if msgs1.shape[0] == 0:
                if msgs2.shape[0] == 0:
                    old_msgs = None
                else:
                    old_msgs = msgs2
            else:
                if msgs2.shape[0] == 0:
                    old_msgs = msgs1
                else:
                    old_msgs = np.vstack((msgs1, msgs2))

            DE3, FA, Khat, Y_detected3, msgs3, Y_decoded3, HARQ_exit, HARQ_curr = scheme.receiver(Y_temp3, DE1 + DE2, nChanlUses_0 + int((2/4) * nChanlUses_harq), old_msgs)

            # Pre-processing for next round

            if HARQ_exit == 1:
                break

            elif HARQ_curr == 1:
                detected_indices = np.array([], dtype=int)

                for i in range(msgs3.shape[0]):
                    detected_indices = np.append(detected_indices, int(bin2dec(msgs3[i, 0:Bf])))

                print("Round 3 status::", DE1 + DE2 + DE3, FA, Khat, msgs3.shape, detected_indices)

                # Pre-processing for next round

                Y_final1 = Y_final1 - Y_decoded3

                Y_final2 = Y_final2 - Y_detected3
            else:
                print("Moving onto next round if available")
                msgs3 = np.array([], dtype=int)

            # if Khat == 0:
            #     resultsFA[Iter, round] = 0
            #     resultsDE[Iter, round] = 0
            # else:
            #     resultsFA[Iter, round] = FA / Khat
            #     resultsDE[Iter, round] = DE3 / K

        elif round == 3:

            print("In round:::", round + 1)

            # --- Preprocessing
            Y_temp4 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, nChanlUses_0 + int((2/4) * nChanlUses_harq)):
                Y_temp4[i, :] = Y_final1[i, :]

            for i in range(nChanlUses_0 + int((2/4) * nChanlUses_harq), nChanlUses_0 + int((3/4) * nChanlUses_harq)):
                Y_temp4[i, :] = Y_final2[i, :]

            # --- Decode
            # Checking the status of previous round detected messages

            if old_msgs is None:
                if msgs3.shape[0] == 0:
                    old_msgs = None
                else:
                    old_msgs = msgs3
            else:
                if msgs3.shape[0] == 0:
                    # old_msgs remain as it is
                    old_msgs = old_msgs
                else:
                    old_msgs = np.vstack((old_msgs, msgs3))


            DE4, FA, Khat, Y_detected4, msgs4, Y_decoded4, HARQ_exit, HARQ_curr = scheme.receiver(Y_temp4, DE1 + DE2 + DE3, nChanlUses_0 + int((3/4) * nChanlUses_harq), old_msgs)

            # Pre-processing for next round

            if HARQ_exit == 1:
                break

            elif HARQ_curr == 1:
                detected_indices = np.array([], dtype=int)

                for i in range(msgs4.shape[0]):
                    detected_indices = np.append(detected_indices, int(bin2dec(msgs4[i, 0:Bf])))

                print("Round 4 status::", DE1 + DE2 + DE3 + DE4, FA, Khat, msgs4.shape, detected_indices)

                # Pre-processing for next round

                Y_final1 = Y_final1 - Y_decoded4

                Y_final2 = Y_final2 - Y_detected4
            else:
                print("Moving onto next round if available")
                msgs4 = np.array([], dtype=int)
            # if Khat == 0:
            #     resultsFA[Iter, round] = 0
            #     resultsDE[Iter, round] = 0
            # else:
            #     resultsFA[Iter, round] = FA / Khat
            #     resultsDE[Iter, round] = DE3 / K


        elif round == 4:

            print("In round:::", round + 1)

            # --- Preprocessing

            Y_temp5 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, nChanlUses_0 + int((3/4) * nChanlUses_harq)):
                Y_temp5[i, :] = Y_final1[i, :]

            for i in range(nChanlUses_0 + int((3/4) * nChanlUses_harq), nChanlUses_0 + int((4/4) * nChanlUses_harq)):
                Y_temp5[i, :] = Y_final2[i, :]

            # --- Decode

            if old_msgs is None:
                if msgs4.shape[0] == 0:
                    old_msgs = None
                else:
                    old_msgs = msgs4
            else:
                if msgs4.shape[0] == 0:
                    # old_msgs remain as it is
                    old_msgs = old_msgs
                else:
                    old_msgs = np.vstack((old_msgs, msgs4))

            DE5, FA, Khat, Y_detected5, msgs5, Y_decoded5, HARQ_exit, HARQ_curr = scheme.receiver(Y_temp5, DE1 + DE2 + DE3 + DE4, nChanlUses_0 + int((4/4) * nChanlUses_harq), old_msgs)

            # Pre-processing for next round

            if HARQ_exit == 1:
                break

            elif HARQ_curr == 1:

                detected_indices = np.array([], dtype=int)

                for i in range(msgs5.shape[0]):
                    detected_indices = np.append(detected_indices, int(bin2dec(msgs5[i, 0:Bf])))

                print("Round 5 status::", DE1 + DE2 + DE3 + DE4 + DE5, FA, Khat, msgs5.shape, detected_indices)

                # Pre-processing for next round

                Y_final1 = Y_final1 - Y_decoded5
                Y_final2 = Y_final2 - Y_detected5
            else:
                print("Moving onto next round if available")
                msgs5 = np.array([], dtype=int)
            # if Khat == 0:
            #     resultsFA[Iter, round] = 0
            #     resultsDE[Iter, round] = 0

            # else:
            #     resultsFA[Iter, round] = FA / Khat
            #     resultsDE[Iter, round] = DE3 / K

#     np.savetxt('resultsDE.out',resultsDE)
#     np.savetxt('resultsFA.out',resultsFA)
#
# missed_det = np.zeros(rounds)
# false_alarm = np.zeros(rounds)
#
# for round in range(rounds):
#     # print('Missed Detection')
#     missed_det[round] = sum(1 - resultsDE[:, round]) / float(nIter)
#     # print(missed_det[round])
#
#     # print('False Alarm')
#     false_alarm[round] = sum(resultsFA[:, round]) / float(nIter)
#     # print(false_alarm[round])
#
#     print("Pe for round:", round , missed_det[round] + false_alarm[round])
#
#
# print("Final simulation result",missed_det + false_alarm)
