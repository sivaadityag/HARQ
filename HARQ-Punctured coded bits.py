# -*- coding: utf-8 -*-
# Python 3.8
import numpy as np
from Fasura import FASURA
import time
from numpy import linalg as L2

np.random.seed(0)
# =========================================== Initialization =========================================== #
# Number of active users
K = 100

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
M = 50
# EbN0dB
EbN0dB = -12

# --- Variance of Noise
sigma2 = 1.0 / ((10 ** (EbN0dB / 10.0)) * B)

# Number of iterations
nIter = 250

# =========================================== Simulation =========================================== #
start = time.time()
print('Number of users: ' + str(K))
print('Number of antennas: ' + str(M))


S = (1 - 2 * np.round(np.random.randint(low=0, high=2, size=(nChanlUses, J)))) + 1j * (1 - 2 * np.round(np.random.randint(low=0, high=2, size=(nChanlUses, J))))

P = S[0:nPilots, :] / np.sqrt(2.0 * nChanlUses)

A = S[nPilots::, :] / np.sqrt(4.0 * nChanlUses)


# --- Generate the channel coefficients
H = (1 / np.sqrt(2)) * (np.random.normal(0, 1, (K, M)) + 1j * np.random.normal(0, 1, (K, M)))

# rounds are zero indexed
rounds = 17

# Performance parameters

detections = np.zeros((nIter, rounds))
packet_energy = np.zeros((nIter, rounds))
total_energy = np.zeros((nIter, 1))

# --- Generate K msgs at once
msgs = np.random.randint(0, 2, (K, B))

# --- HARQ channel uses

# First point to start the process: Half the codeword

nChanlUses_round0 = int(3*(L * (nc/2)/4))
nChanlUses_harq = int(1*(L * (nc/2)/4))
nChanlUses_0 = nPilots + nChanlUses_round0

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

            # Keep two copies of the entire signal

            Y_final1 = Y.copy()
            Y_final2 = Y.copy()

            #Performance parameter

            total_energy[Iter][round] = L2.norm(Y)**2

            # print("In round:::", round + 1)

            # --- Preprocessing
            Y_temp1 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, nChanlUses_0):
                Y_temp1[i, :] = Y_final1[i, :]

            # --- Decode

            DE1, FA, Khat, Y_detected1, msgs1, Y_decoded1, HARQ_exit, HARQ_curr = scheme.receiver(Y_temp1, 0, nChanlUses_0, None)

            # All detections complete: No more energy left
            if HARQ_exit == 1:
                # print("All possible detections complete")
                break

            # Current round had successful detections
            elif HARQ_curr == 1:
                # detected_indices = np.array([], dtype=int)
                #
                # for i in range(msgs1.shape[0]):
                #     detected_indices = np.append(detected_indices, int(bin2dec(msgs1[i, 0:Bf])))

                # print("Round 1 status::", DE1, FA, Khat, msgs1.shape)

                # Pre-processing for next round
                Y_final1 = Y_final1 - Y_decoded1
                Y_final2 = Y_final2 - Y_detected1

            # Current round failed (Unable to detect any new users)
            else:
                # print("No new detections: Moving onto next round")
                msgs1 = np.array([], dtype=int)

            # Performance parameters

            detections[Iter][round] = DE1
            packet_energy[Iter][round] = L2.norm(Y_temp1)**2

        elif round == 1:

            # print("In round:::", round + 1)

            # --- Preprocessing
            Y_temp2 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, nChanlUses_0):
                Y_temp2[i, :] = Y_final1[i, :]

            for i in range(nChanlUses_0, nChanlUses_0 + int((1/16) * nChanlUses_harq)):
                Y_temp2[i, :] = Y_final2[i, :]

            # --- Decode

            # Check detected msgs from previous rounds

            if msgs1.shape[0] == 0:
                old_msgs = None
            else:
                old_msgs = msgs1

            DE2, FA, Khat, Y_detected2, msgs2, Y_decoded2, HARQ_exit, HARQ_curr = scheme.receiver(Y_temp2, DE1, nChanlUses_0 + int((1/16) * nChanlUses_harq) , old_msgs)

            # All detections complete: No more energy left
            if HARQ_exit == 1:
                # print("All possible detections complete")
                break

            # Current round had successful detections
            elif HARQ_curr == 1:

                # detected_indices = np.array([], dtype=int)
                #
                # for i in range(msgs2.shape[0]):
                #     detected_indices = np.append(detected_indices, int(bin2dec(msgs2[i, 0:Bf])))

                # print("Round 2 status::", DE1 + DE2, FA, Khat, msgs2.shape)

                # Pre-processing for next round

                Y_final1 = Y_final1 - Y_decoded2
                Y_final2 = Y_final2 - Y_detected2

            # Current round failed (Unable to detect any new users)
            else:
                # print("No new detections: Moving onto next round")
                msgs2 = np.array([], dtype=int)

            # Performance parameters

            detections[Iter][round] = DE1 + DE2
            packet_energy[Iter][round] = L2.norm(Y_temp2[nChanlUses_0::, :]) ** 2

        elif round == 2:

            # print("In round:::", round + 1)

            # --- Preprocessing
            Y_temp3 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, nChanlUses_0 + int((1/16) * nChanlUses_harq)):
                Y_temp3[i, :] = Y_final1[i, :]

            for i in range(nChanlUses_0 + int((1/16) * nChanlUses_harq), nChanlUses_0 + int((2/16) * nChanlUses_harq)):
                Y_temp3[i, :] = Y_final2[i, :]

            # --- Decode

            # Check detected msgs from previous rounds

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

            DE3, FA, Khat, Y_detected3, msgs3, Y_decoded3, HARQ_exit, HARQ_curr = scheme.receiver(Y_temp3, DE1 + DE2, nChanlUses_0 + int((2/16) * nChanlUses_harq), old_msgs)

            # All detections complete: No more energy left
            if HARQ_exit == 1:
                # print("All possible detections complete")
                break

            # Current round had successful detections
            elif HARQ_curr == 1:
                # detected_indices = np.array([], dtype=int)
                #
                # for i in range(msgs3.shape[0]):
                #     detected_indices = np.append(detected_indices, int(bin2dec(msgs3[i, 0:Bf])))

                # print("Round 3 status::", DE1 + DE2 + DE3, FA, Khat, msgs3.shape)

                # Pre-processing for next round

                Y_final1 = Y_final1 - Y_decoded3
                Y_final2 = Y_final2 - Y_detected3

            # Current round failed (Unable to detect any new users)
            else:
                # print("No new detections: Moving onto next round")
                msgs3 = np.array([], dtype=int)

            # Performance parameters

            detections[Iter][round] = DE1 + DE2 + DE3
            packet_energy[Iter][round] = L2.norm(Y_temp3[nChanlUses_0 + int((1/16) * nChanlUses_harq)::, :]) ** 2

        elif round == 3:

            # print("In round:::", round + 1)

            # --- Preprocessing
            Y_temp4 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, nChanlUses_0 + int((2/16) * nChanlUses_harq)):
                Y_temp4[i, :] = Y_final1[i, :]

            for i in range(nChanlUses_0 + int((2/16) * nChanlUses_harq), nChanlUses_0 + int((3/16) * nChanlUses_harq)):
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


            DE4, FA, Khat, Y_detected4, msgs4, Y_decoded4, HARQ_exit, HARQ_curr = scheme.receiver(Y_temp4, DE1 + DE2 + DE3, nChanlUses_0 + int((3/16) * nChanlUses_harq), old_msgs)

            # All detections complete: No more energy left
            if HARQ_exit == 1:
                # print("All possible detections complete")
                break

            # Current round had successful detections
            elif HARQ_curr == 1:
                # detected_indices = np.array([], dtype=int)
                #
                # for i in range(msgs4.shape[0]):
                #     detected_indices = np.append(detected_indices, int(bin2dec(msgs4[i, 0:Bf])))

                # print("Round 4 status::", DE1 + DE2 + DE3 + DE4, FA, Khat, msgs4.shape)

                # Pre-processing for next round
                Y_final1 = Y_final1 - Y_decoded4
                Y_final2 = Y_final2 - Y_detected4

            # Current round failed (Unable to detect any new users)
            else:
                # print("No new detections: Moving onto next round")
                msgs4 = np.array([], dtype=int)

            # Performance parameters

            detections[Iter][round] = DE1 + DE2 + DE3 + DE4
            packet_energy[Iter][round] = L2.norm(Y_temp4[nChanlUses_0 + int((2 / 16) * nChanlUses_harq)::, :]) ** 2

        elif round == 4:

            # print("In round:::", round + 1)

            # --- Preprocessing

            Y_temp5 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, nChanlUses_0 + int((3/16) * nChanlUses_harq)):
                Y_temp5[i, :] = Y_final1[i, :]

            for i in range(nChanlUses_0 + int((3/16) * nChanlUses_harq), nChanlUses_0 + int((4/16) * nChanlUses_harq)):
                Y_temp5[i, :] = Y_final2[i, :]

            # --- Decode

            # Checking the status of previous round detected messages

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

            DE5, FA, Khat, Y_detected5, msgs5, Y_decoded5, HARQ_exit, HARQ_curr = scheme.receiver(Y_temp5, DE1 + DE2 + DE3 + DE4, nChanlUses_0 + int((4/16) * nChanlUses_harq), old_msgs)

            # All detections complete: No more energy left
            if HARQ_exit == 1:
                # print("All possible detections complete")
                break

            # Current round had successful detections
            elif HARQ_curr == 1:
                #
                # detected_indices = np.array([], dtype=int)
                #
                # for i in range(msgs5.shape[0]):
                #     detected_indices = np.append(detected_indices, int(bin2dec(msgs5[i, 0:Bf])))

                # print("Round 5 status::", DE1 + DE2 + DE3 + DE4 + DE5, FA, Khat, msgs5.shape)

                # Pre-processing for next round

                Y_final1 = Y_final1 - Y_decoded5
                Y_final2 = Y_final2 - Y_detected5

            # Current round failed (Unable to detect any new users)
            else:
                # print("No new detections: Moving onto next round")
                msgs5 = np.array([], dtype=int)

            # Performance parameters

            detections[Iter][round] = DE1 + DE2 + DE3 + DE4 + DE5
            packet_energy[Iter][round] = L2.norm(Y_temp5[nChanlUses_0 + int((3 / 16) * nChanlUses_harq)::, :]) ** 2

        elif round == 5:

            # print("In round:::", round + 1)

            # --- Preprocessing

            Y_temp6 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, nChanlUses_0 + int((4/16) * nChanlUses_harq)):
                Y_temp6[i, :] = Y_final1[i, :]

            for i in range(nChanlUses_0 + int((4/16) * nChanlUses_harq), nChanlUses_0 + int((5/16) * nChanlUses_harq)):
                Y_temp6[i, :] = Y_final2[i, :]

            # --- Decode

            # Checking the status of previous round detected messages

            if old_msgs is None:
                if msgs5.shape[0] == 0:
                    old_msgs = None
                else:
                    old_msgs = msgs5
            else:
                if msgs5.shape[0] == 0:
                    # old_msgs remain as it is
                    old_msgs = old_msgs
                else:
                    old_msgs = np.vstack((old_msgs, msgs5))

            DE6, FA, Khat, Y_detected6, msgs6, Y_decoded6, HARQ_exit, HARQ_curr = scheme.receiver(Y_temp6, DE1 + DE2 + DE3 + DE4 + DE5, nChanlUses_0 + int((5/16) * nChanlUses_harq), old_msgs)

            # All detections complete: No more energy left
            if HARQ_exit == 1:
                # print("All possible detections complete")
                break

            # Current round had successful detections
            elif HARQ_curr == 1:

                # detected_indices = np.array([], dtype=int)
                #
                # # for i in range(msgs6.shape[0]):
                #     detected_indices = np.append(detected_indices, int(bin2dec(msgs6[i, 0:Bf])))

                # print("Round 6 status::", DE1 + DE2 + DE3 + DE4 + DE5 + DE6, FA, Khat, msgs6.shape)

                # Pre-processing for next round

                Y_final1 = Y_final1 - Y_decoded6
                Y_final2 = Y_final2 - Y_detected6

            # Current round failed (Unable to detect any new users)
            else:
                # print("No new detections: Moving onto next round")
                msgs6 = np.array([], dtype=int)

            # Performance parameters

            detections[Iter][round] = DE1 + DE2 + DE3 + DE4 + DE5 + DE6
            packet_energy[Iter][round] = L2.norm(Y_temp6[nChanlUses_0 + int((4 / 16) * nChanlUses_harq)::, :]) ** 2

        elif round == 6:

            # print("In round:::", round + 1)

            # --- Preprocessing

            Y_temp7 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, nChanlUses_0 + int((5 / 16) * nChanlUses_harq)):
                Y_temp7[i, :] = Y_final1[i, :]

            for i in range(nChanlUses_0 + int((5 / 16) * nChanlUses_harq), nChanlUses_0 + int((6 / 16) * nChanlUses_harq)):
                Y_temp7[i, :] = Y_final2[i, :]

            # --- Decode

            # Checking the status of previous round detected messages

            if old_msgs is None:
                if msgs6.shape[0] == 0:
                    old_msgs = None
                else:
                    old_msgs = msgs6
            else:
                if msgs6.shape[0] == 0:
                    # old_msgs remain as it is
                    old_msgs = old_msgs
                else:
                    old_msgs = np.vstack((old_msgs, msgs6))

            DE7, FA, Khat, Y_detected7, msgs7, Y_decoded7, HARQ_exit, HARQ_curr = scheme.receiver(Y_temp6, DE1 + DE2 + DE3 + DE4 + DE5 + DE6, nChanlUses_0 + int((6 / 16) * nChanlUses_harq), old_msgs)

            # All detections complete: No more energy left
            if HARQ_exit == 1:
                # print("All possible detections complete")
                break

            # Current round had successful detections
            elif HARQ_curr == 1:

                # detected_indices = np.array([], dtype=int)
                #
                # for i in range(msgs6.shape[0]):
                #     detected_indices = np.append(detected_indices, int(bin2dec(msgs6[i, 0:Bf])))

                # print("Round 7 status::", DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7, FA, Khat, msgs7.shape)

                # Pre-processing for next round

                Y_final1 = Y_final1 - Y_decoded7
                Y_final2 = Y_final2 - Y_detected7

            # Current round failed (Unable to detect any new users)
            else:
                # print("No new detections: Moving onto next round")
                msgs7 = np.array([], dtype=int)

            # Performance parameters

            detections[Iter][round] = DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7
            packet_energy[Iter][round] = L2.norm(Y_temp7[nChanlUses_0 + int((5 / 16) * nChanlUses_harq)::, :]) ** 2

        elif round == 7:

            # print("In round:::", round + 1)

            # --- Preprocessing

            Y_temp8 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, nChanlUses_0 + int((6 / 16) * nChanlUses_harq)):
                Y_temp8[i, :] = Y_final1[i, :]

            for i in range(nChanlUses_0 + int((6 / 16) * nChanlUses_harq), nChanlUses_0 + int((7 / 16) * nChanlUses_harq)):
                Y_temp8[i, :] = Y_final2[i, :]

            # --- Decode

            # Checking the status of previous round detected messages

            if old_msgs is None:
                if msgs7.shape[0] == 0:
                    old_msgs = None
                else:
                    old_msgs = msgs7
            else:
                if msgs7.shape[0] == 0:
                    # old_msgs remain as it is
                    old_msgs = old_msgs
                else:
                    old_msgs = np.vstack((old_msgs, msgs7))

            DE8, FA, Khat, Y_detected8, msgs8, Y_decoded8, HARQ_exit, HARQ_curr = scheme.receiver(Y_temp8, DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7, nChanlUses_0 + int((7 / 16) * nChanlUses_harq), old_msgs)

            # All detections complete: No more energy left
            if HARQ_exit == 1:
                # print("All possible detections complete")
                break

            # Current round had successful detections
            elif HARQ_curr == 1:

                # detected_indices = np.array([], dtype=int)
                #
                # for i in range(msgs6.shape[0]):
                #     detected_indices = np.append(detected_indices, int(bin2dec(msgs6[i, 0:Bf])))

                # print("Round 8 status::", DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8, FA, Khat, msgs8.shape)

                # Pre-processing for next round

                Y_final1 = Y_final1 - Y_decoded8
                Y_final2 = Y_final2 - Y_detected8

            # Current round failed (Unable to detect any new users)
            else:
                # print("No new detections: Moving onto next round")
                msgs8 = np.array([], dtype=int)

            # Performance parameters

            detections[Iter][round] = DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8
            packet_energy[Iter][round] = L2.norm(Y_temp8[nChanlUses_0 + int((6 / 16) * nChanlUses_harq)::, :]) ** 2

        elif round == 8:

            # print("In round:::", round + 1)

            # --- Preprocessing

            Y_temp9 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, nChanlUses_0 + int((7 / 16) * nChanlUses_harq)):
                Y_temp9[i, :] = Y_final1[i, :]

            for i in range(nChanlUses_0 + int((7 / 16) * nChanlUses_harq), nChanlUses_0 + int((8 / 16) * nChanlUses_harq)):
                Y_temp9[i, :] = Y_final2[i, :]

            # --- Decode

            # Checking the status of previous round detected messages

            if old_msgs is None:
                if msgs8.shape[0] == 0:
                    old_msgs = None
                else:
                    old_msgs = msgs8
            else:
                if msgs8.shape[0] == 0:
                    # old_msgs remain as it is
                    old_msgs = old_msgs
                else:
                    old_msgs = np.vstack((old_msgs, msgs8))

            DE9, FA, Khat, Y_detected9, msgs9, Y_decoded9, HARQ_exit, HARQ_curr = scheme.receiver(Y_temp9, DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8, nChanlUses_0 + int((8 / 16) * nChanlUses_harq), old_msgs)

            # All detections complete: No more energy left
            if HARQ_exit == 1:
                # print("All possible detections complete")
                break

            # Current round had successful detections
            elif HARQ_curr == 1:

                # detected_indices = np.array([], dtype=int)
                #
                # for i in range(msgs6.shape[0]):
                #     detected_indices = np.append(detected_indices, int(bin2dec(msgs6[i, 0:Bf])))

                # print("Round 9 status::", DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9, FA, Khat, msgs9.shape)

                # Pre-processing for next round

                Y_final1 = Y_final1 - Y_decoded9
                Y_final2 = Y_final2 - Y_detected9

            # Current round failed (Unable to detect any new users)
            else:
                # print("No new detections: Moving onto next round")
                msgs9 = np.array([], dtype=int)

            # Performance parameters

            detections[Iter][round] = DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9
            packet_energy[Iter][round] = L2.norm(Y_temp9[nChanlUses_0 + int((7 / 16) * nChanlUses_harq)::, :]) ** 2

        elif round == 9:

            # print("In round:::", round + 1)

            # --- Preprocessing

            Y_temp10 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, nChanlUses_0 + int((8 / 16) * nChanlUses_harq)):
                Y_temp10[i, :] = Y_final1[i, :]

            for i in range(nChanlUses_0 + int((8 / 16) * nChanlUses_harq), nChanlUses_0 + int((9 / 16) * nChanlUses_harq)):
                Y_temp10[i, :] = Y_final2[i, :]

            # --- Decode

            # Checking the status of previous round detected messages

            if old_msgs is None:
                if msgs9.shape[0] == 0:
                    old_msgs = None
                else:
                    old_msgs = msgs9
            else:
                if msgs9.shape[0] == 0:
                    # old_msgs remain as it is
                    old_msgs = old_msgs
                else:
                    old_msgs = np.vstack((old_msgs, msgs9))

            DE10, FA, Khat, Y_detected10, msgs10, Y_decoded10, HARQ_exit, HARQ_curr = scheme.receiver(Y_temp10, DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9, nChanlUses_0 + int((9 / 16) * nChanlUses_harq), old_msgs)

            # All detections complete: No more energy left
            if HARQ_exit == 1:
                # print("All possible detections complete")
                break

            # Current round had successful detections
            elif HARQ_curr == 1:

                # detected_indices = np.array([], dtype=int)
                #
                # for i in range(msgs6.shape[0]):
                #     detected_indices = np.append(detected_indices, int(bin2dec(msgs6[i, 0:Bf])))

                # print("Round 10 status::", DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9 + DE10, FA, Khat, msgs10.shape)

                # Pre-processing for next round

                Y_final1 = Y_final1 - Y_decoded10
                Y_final2 = Y_final2 - Y_detected10

            # Current round failed (Unable to detect any new users)
            else:
                # print("No new detections: Moving onto next round")
                msgs10 = np.array([], dtype=int)

            # Performance parameters

            detections[Iter][round] = DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9 + DE10
            packet_energy[Iter][round] = L2.norm(Y_temp10[nChanlUses_0 + int((8 / 16) * nChanlUses_harq)::, :]) ** 2

        elif round == 10:

            # print("In round:::", round + 1)

            # --- Preprocessing

            Y_temp11 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, nChanlUses_0 + int((9 / 16) * nChanlUses_harq)):
                Y_temp11[i, :] = Y_final1[i, :]

            for i in range(nChanlUses_0 + int((9 / 16) * nChanlUses_harq), nChanlUses_0 + int((10 / 16) * nChanlUses_harq)):
                Y_temp11[i, :] = Y_final2[i, :]

            # --- Decode

            # Checking the status of previous round detected messages

            if old_msgs is None:
                if msgs10.shape[0] == 0:
                    old_msgs = None
                else:
                    old_msgs = msgs10
            else:
                if msgs10.shape[0] == 0:
                    # old_msgs remain as it is
                    old_msgs = old_msgs
                else:
                    old_msgs = np.vstack((old_msgs, msgs10))

            DE11, FA, Khat, Y_detected11, msgs11, Y_decoded11, HARQ_exit, HARQ_curr = scheme.receiver(Y_temp11, DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9 + DE10, nChanlUses_0 + int((10 / 16) * nChanlUses_harq), old_msgs)

            # All detections complete: No more energy left
            if HARQ_exit == 1:
                # print("All possible detections complete")
                break

            # Current round had successful detections
            elif HARQ_curr == 1:

                # detected_indices = np.array([], dtype=int)
                #
                # for i in range(msgs6.shape[0]):
                #     detected_indices = np.append(detected_indices, int(bin2dec(msgs6[i, 0:Bf])))

                # print("Round 11 status::", DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9 + DE10 + DE11, FA, Khat, msgs11.shape)

                # Pre-processing for next round

                Y_final1 = Y_final1 - Y_decoded11
                Y_final2 = Y_final2 - Y_detected11

            # Current round failed (Unable to detect any new users)
            else:
                # print("No new detections: Moving onto next round")
                msgs11 = np.array([], dtype=int)

            # Performance parameters

            detections[Iter][round] = DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9 + DE10 + DE11
            packet_energy[Iter][round] = L2.norm(Y_temp11[nChanlUses_0 + int((9 / 16) * nChanlUses_harq)::, :]) ** 2

        elif round == 11:

            # print("In round:::", round + 1)

            # --- Preprocessing

            Y_temp12 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, nChanlUses_0 + int((10 / 16) * nChanlUses_harq)):
                Y_temp12[i, :] = Y_final1[i, :]

            for i in range(nChanlUses_0 + int((10 / 16) * nChanlUses_harq), nChanlUses_0 + int((11 / 16) * nChanlUses_harq)):
                Y_temp12[i, :] = Y_final2[i, :]

            # --- Decode

            # Checking the status of previous round detected messages

            if old_msgs is None:
                if msgs11.shape[0] == 0:
                    old_msgs = None
                else:
                    old_msgs = msgs11
            else:
                if msgs11.shape[0] == 0:
                    # old_msgs remain as it is
                    old_msgs = old_msgs
                else:
                    old_msgs = np.vstack((old_msgs, msgs11))

            DE12, FA, Khat, Y_detected12, msgs12, Y_decoded12, HARQ_exit, HARQ_curr = scheme.receiver(Y_temp12, DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9 + DE10 + DE11, nChanlUses_0 + int((11 / 16) * nChanlUses_harq), old_msgs)

            # All detections complete: No more energy left
            if HARQ_exit == 1:
                # print("All possible detections complete")
                break

            # Current round had successful detections
            elif HARQ_curr == 1:

                # detected_indices = np.array([], dtype=int)
                #
                # for i in range(msgs6.shape[0]):
                #     detected_indices = np.append(detected_indices, int(bin2dec(msgs6[i, 0:Bf])))

                # print("Round 12 status::", DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9 + DE10 + DE11 + DE12, FA, Khat, msgs12.shape)

                # Pre-processing for next round

                Y_final1 = Y_final1 - Y_decoded12
                Y_final2 = Y_final2 - Y_detected12

            # Current round failed (Unable to detect any new users)
            else:
                # print("No new detections: Moving onto next round")
                msgs12 = np.array([], dtype=int)

            # Performance parameters

            detections[Iter][round] = DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9 + DE10 + DE11 + DE12
            packet_energy[Iter][round] = L2.norm(Y_temp12[nChanlUses_0 + int((10 / 16) * nChanlUses_harq)::, :]) ** 2

        elif round == 12:

            # print("In round:::", round + 1)

            # --- Preprocessing

            Y_temp13 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, nChanlUses_0 + int((11 / 16) * nChanlUses_harq)):
                Y_temp13[i, :] = Y_final1[i, :]

            for i in range(nChanlUses_0 + int((11 / 16) * nChanlUses_harq), nChanlUses_0 + int((12 / 16) * nChanlUses_harq)):
                Y_temp13[i, :] = Y_final2[i, :]

            # --- Decode

            # Checking the status of previous round detected messages

            if old_msgs is None:
                if msgs12.shape[0] == 0:
                    old_msgs = None
                else:
                    old_msgs = msgs12
            else:
                if msgs12.shape[0] == 0:
                    # old_msgs remain as it is
                    old_msgs = old_msgs
                else:
                    old_msgs = np.vstack((old_msgs, msgs12))

            DE13, FA, Khat, Y_detected13, msgs13, Y_decoded13, HARQ_exit, HARQ_curr = scheme.receiver(Y_temp13,
                                                                                                      DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9 + DE10 + DE11 + DE12,
                                                                                                      nChanlUses_0 + int((12 / 16) * nChanlUses_harq),
                                                                                                      old_msgs)

            # All detections complete: No more energy left
            if HARQ_exit == 1:
                # print("All possible detections complete")
                break

            # Current round had successful detections
            elif HARQ_curr == 1:

                # detected_indices = np.array([], dtype=int)
                #
                # for i in range(msgs6.shape[0]):
                #     detected_indices = np.append(detected_indices, int(bin2dec(msgs6[i, 0:Bf])))

                # print("Round 12 status::", DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9 + DE10 + DE11 + DE12 + DE13, FA, Khat, msgs13.shape)

                # Pre-processing for next round

                Y_final1 = Y_final1 - Y_decoded13
                Y_final2 = Y_final2 - Y_detected13

            # Current round failed (Unable to detect any new users)
            else:
                # print("No new detections: Moving onto next round")
                msgs13 = np.array([], dtype=int)

            # Performance parameters

            detections[Iter][round] = DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9 + DE10 + DE11 + DE12 + DE13
            packet_energy[Iter][round] = L2.norm(Y_temp13[nChanlUses_0 + int((11 / 16) * nChanlUses_harq)::, :]) ** 2

        elif round == 13:

            # print("In round:::", round + 1)

            # --- Preprocessing

            Y_temp14 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, nChanlUses_0 + int((12 / 16) * nChanlUses_harq)):
                Y_temp14[i, :] = Y_final1[i, :]

            for i in range(nChanlUses_0 + int((12 / 16) * nChanlUses_harq), nChanlUses_0 + int((13 / 16) * nChanlUses_harq)):
                Y_temp14[i, :] = Y_final2[i, :]

            # --- Decode

            # Checking the status of previous round detected messages

            if old_msgs is None:
                if msgs13.shape[0] == 0:
                    old_msgs = None
                else:
                    old_msgs = msgs13
            else:
                if msgs13.shape[0] == 0:
                    # old_msgs remain as it is
                    old_msgs = old_msgs
                else:
                    old_msgs = np.vstack((old_msgs, msgs13))

            DE14, FA, Khat, Y_detected14, msgs14, Y_decoded14, HARQ_exit, HARQ_curr = scheme.receiver(Y_temp14,
                                                                                                      DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9 + DE10 + DE11 + DE12 + DE13,
                                                                                                      nChanlUses_0 + int((13 / 16) * nChanlUses_harq),
                                                                                                      old_msgs)

            # All detections complete: No more energy left
            if HARQ_exit == 1:
                # print("All possible detections complete")
                break

            # Current round had successful detections
            elif HARQ_curr == 1:

                # detected_indices = np.array([], dtype=int)
                #
                # for i in range(msgs6.shape[0]):
                #     detected_indices = np.append(detected_indices, int(bin2dec(msgs6[i, 0:Bf])))

                # print("Round 14 status::", DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9 + DE10 + DE11 + DE12 + DE13 + DE14, FA, Khat, msgs14.shape)

                # Pre-processing for next round

                Y_final1 = Y_final1 - Y_decoded14
                Y_final2 = Y_final2 - Y_detected14

            # Current round failed (Unable to detect any new users)
            else:
                # print("No new detections: Moving onto next round")
                msgs14 = np.array([], dtype=int)

            # Performance parameters

            detections[Iter][round] = DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9 + DE10 + DE11 + DE12 + DE13 +DE14
            packet_energy[Iter][round] = L2.norm(Y_temp14[nChanlUses_0 + int((12 / 16) * nChanlUses_harq)::, :]) ** 2

        elif round == 14:

            # print("In round:::", round + 1)

            # --- Preprocessing

            Y_temp15 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, nChanlUses_0 + int((13 / 16) * nChanlUses_harq)):
                Y_temp15[i, :] = Y_final1[i, :]

            for i in range(nChanlUses_0 + int((13 / 16) * nChanlUses_harq), nChanlUses_0 + int((14 / 16) * nChanlUses_harq)):
                Y_temp15[i, :] = Y_final2[i, :]

            # --- Decode

            # Checking the status of previous round detected messages

            if old_msgs is None:
                if msgs14.shape[0] == 0:
                    old_msgs = None
                else:
                    old_msgs = msgs14
            else:
                if msgs14.shape[0] == 0:
                    # old_msgs remain as it is
                    old_msgs = old_msgs
                else:
                    old_msgs = np.vstack((old_msgs, msgs14))

            DE15, FA, Khat, Y_detected15, msgs15, Y_decoded15, HARQ_exit, HARQ_curr = scheme.receiver(Y_temp15,
                                                                                                      DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9 + DE10 + DE11 + DE12 + DE13 + DE14,
                                                                                                      nChanlUses_0 + int((14 / 16) * nChanlUses_harq),
                                                                                                      old_msgs)

            # All detections complete: No more energy left
            if HARQ_exit == 1:
                # print("All possible detections complete")
                break

            # Current round had successful detections
            elif HARQ_curr == 1:

                # detected_indices = np.array([], dtype=int)
                #
                # for i in range(msgs6.shape[0]):
                #     detected_indices = np.append(detected_indices, int(bin2dec(msgs6[i, 0:Bf])))

                # print("Round 15 status::", DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9 + DE10 + DE11 + DE12 + DE13 + DE14 + DE15, FA,                      Khat, msgs15.shape)

                # Pre-processing for next round

                Y_final1 = Y_final1 - Y_decoded15
                Y_final2 = Y_final2 - Y_detected15

            # Current round failed (Unable to detect any new users)
            else:
                # print("No new detections: Moving onto next round")
                msgs15 = np.array([], dtype=int)

            # Performance parameters

            detections[Iter][round] = DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9 + DE10 + DE11 + DE12 + DE13 + DE14 + DE15
            packet_energy[Iter][round] = L2.norm(Y_temp15[nChanlUses_0 + int((13 / 16) * nChanlUses_harq)::, :]) ** 2

        elif round == 15:

            # print("In round:::", round + 1)

            # --- Preprocessing

            Y_temp16 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, nChanlUses_0 + int((14 / 16) * nChanlUses_harq)):
                Y_temp16[i, :] = Y_final1[i, :]

            for i in range(nChanlUses_0 + int((14 / 16) * nChanlUses_harq), nChanlUses_0 + int((15 / 16) * nChanlUses_harq)):
                Y_temp16[i, :] = Y_final2[i, :]

            # --- Decode

            # Checking the status of previous round detected messages

            if old_msgs is None:
                if msgs15.shape[0] == 0:
                    old_msgs = None
                else:
                    old_msgs = msgs15
            else:
                if msgs15.shape[0] == 0:
                    # old_msgs remain as it is
                    old_msgs = old_msgs
                else:
                    old_msgs = np.vstack((old_msgs, msgs15))

            DE16, FA, Khat, Y_detected16, msgs16, Y_decoded16, HARQ_exit, HARQ_curr = scheme.receiver(Y_temp16,
                                                                                                      DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9 + DE10 + DE11 + DE12 + DE13 + DE14 + DE15,
                                                                                                      nChanlUses_0 + int((15 / 16) * nChanlUses_harq),
                                                                                                      old_msgs)

            # All detections complete: No more energy left
            if HARQ_exit == 1:
                # print("All possible detections complete")
                break

            # Current round had successful detections
            elif HARQ_curr == 1:

                # detected_indices = np.array([], dtype=int)
                #
                # for i in range(msgs6.shape[0]):
                #     detected_indices = np.append(detected_indices, int(bin2dec(msgs6[i, 0:Bf])))

                # print("Round 16 status::", DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9 + DE10 + DE11 + DE12 + DE13 + DE14 + DE15 + DE16, FA, Khat, msgs16.shape)

                # Pre-processing for next round

                Y_final1 = Y_final1 - Y_decoded16
                Y_final2 = Y_final2 - Y_detected16

            # Current round failed (Unable to detect any new users)
            else:
                # print("No new detections: Moving onto next round")
                msgs16 = np.array([], dtype=int)

            # Performance parameters

            detections[Iter][round] = DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9 + DE10 + DE11 + DE12 + DE13 + DE14 + DE15 + DE16
            packet_energy[Iter][round] = L2.norm(Y_temp16[nChanlUses_0 + int((14 / 16) * nChanlUses_harq)::, :]) ** 2

        elif round == 16:

            # print("In round:::", round + 1)

            # --- Preprocessing

            Y_temp17 = np.zeros((nChanlUses, M), dtype=complex)

            for i in range(0, nChanlUses_0 + int((15 / 16) * nChanlUses_harq)):
                Y_temp17[i, :] = Y_final1[i, :]

            for i in range(nChanlUses_0 + int((15 / 16) * nChanlUses_harq), nChanlUses_0 + int((16 / 16) * nChanlUses_harq)):
                Y_temp17[i, :] = Y_final2[i, :]

            # --- Decode

            # Checking the status of previous round detected messages

            if old_msgs is None:
                if msgs16.shape[0] == 0:
                    old_msgs = None
                else:
                    old_msgs = msgs16
            else:
                if msgs16.shape[0] == 0:
                    # old_msgs remain as it is
                    old_msgs = old_msgs
                else:
                    old_msgs = np.vstack((old_msgs, msgs16))

            DE17, FA, Khat, Y_detected17, msgs17, Y_decoded17, HARQ_exit, HARQ_curr = scheme.receiver(Y_temp17,
                                                                                                      DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9 + DE10 + DE11 + DE12 + DE13 + DE14 + DE15 + DE16,
                                                                                                      nChanlUses_0 + int((16 / 16) * nChanlUses_harq),
                                                                                                      old_msgs)

            # All detections complete: No more energy left
            if HARQ_exit == 1:
                # print("All possible detections complete")
                break

            # Current round had successful detections
            elif HARQ_curr == 1:

                # detected_indices = np.array([], dtype=int)
                #
                # for i in range(msgs6.shape[0]):
                #     detected_indices = np.append(detected_indices, int(bin2dec(msgs6[i, 0:Bf])))

                # print("Round 17 status::", DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9 + DE10 + DE11 + DE12 + DE13 + DE14 + DE15 + DE16 + DE17, FA, Khat, msgs17.shape)

                # Pre-processing for next round

                Y_final1 = Y_final1 - Y_decoded17
                Y_final2 = Y_final2 - Y_detected17

            # Current round failed (Unable to detect any new users)
            else:
                # print("No new detections: Moving onto next round")
                msgs17 = np.array([], dtype=int)

            # Performance parameters

            detections[Iter][round] = DE1 + DE2 + DE3 + DE4 + DE5 + DE6 + DE7 + DE8 + DE9 + DE10 + DE11 + DE12 + DE13 + DE14 + DE15 + DE16 + DE17
            packet_energy[Iter][round] = L2.norm(Y_temp17[nChanlUses_0 + int((15 / 16) * nChanlUses_harq)::, :]) ** 2

    np.savetxt('Detections.out', detections)
    np.savetxt('Packet_energy.out', packet_energy)
    np.savetxt('Full_packet_energy.out', total_energy)
