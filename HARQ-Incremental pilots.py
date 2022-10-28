# -*- coding: utf-8 -*-
# Python 3.8
import numpy as np
from Fasura import FASURA
import time


def uplink_round(K, nPilots, B, Bf, L, nc, nL, M, sigma2, H, nChanlUses):
    np.random.seed(0)

    S = (1 - 2 * np.round(np.random.randint(low=0, high=2, size=(nChanlUses, J)))) + 1j * (
            1 - 2 * np.round(np.random.randint(low=0, high=2, size=(nChanlUses, J))))

    P = S[0:nPilots, :] / np.sqrt(2.0 * nChanlUses)

    A = S[nPilots::, :] / np.sqrt(4.0 * nChanlUses)

    # --- Create a FASURA object
    scheme = FASURA(K, nPilots, B, Bf, L, nc, nL, M, sigma2, H, P, A, nChanlUses)

    # --- Encode the data
    XH = scheme.transmitter(msgs, H)

    # --- Generate the noiseww
    N = (1 / np.sqrt(2)) * (
            np.random.normal(0, np.sqrt(sigma2), (nChanlUses, M)) + 1j * np.random.normal(0, np.sqrt(sigma2),
                                                                                          (nChanlUses, M)))
    # --- Received Signal

    Y = XH + N

    # --- Decode

    print("Shape of old signal",Y.shape)
    DE, FA, Khat = scheme.receiver(Y)
    if Khat == 0:
        resultsFA = 0
        resultsDE = 0
    else:
        resultsFA = FA / Khat
        resultsDE = DE / K

    # print('Missed Detection')
    missed_det = 1 - resultsDE
    print(missed_det)

    # print('False Alarm')
    false_alarm = resultsFA
    print(false_alarm)

    return missed_det + false_alarm, Y[nPilots::,:], A


def uplink_pilot_round(K, nPilots, B, Bf, L, nc, nL, M, sigma2, H, nChanlUses, Y_msgPart, A, msgs):
    np.random.seed(0)

    S = (1 - 2 * np.round(np.random.randint(low=0, high=2, size=(nChanlUses, J)))) + 1j * (
         1 - 2 * np.round(np.random.randint(low=0, high=2, size=(nChanlUses, J))))

    P = S[0:nPilots, :] / np.sqrt(2.0 * nChanlUses)

    # --- Create a FASURA object
    scheme = FASURA(K, nPilots, B, Bf, L, nc, nL, M, sigma2, H, P, A, int(nChanlUses))

    # --- Generate the noise
    N = (1 / np.sqrt(2)) * (
            np.random.normal(0, np.sqrt(sigma2), (nPilots, M)) + 1j * np.random.normal(0, np.sqrt(sigma2),
                                                                                       (nPilots, M)))
    # --- Transmit new pilots

    Y_newpilot = scheme.pilot_transmitter(msgs, H) + N
    print("Shape of new pilots",Y_newpilot.shape)


    # --- Combine the signal and decode
    # --- Add the new matrix to the output signal

    Y = np.vstack((Y_newpilot, Y_msgPart))
    print("Shape of new signal",Y.shape)
    DE, FA, Khat = scheme.receiver(Y)

    if Khat == 0:
        resultsFA = 0
        resultsDE = 0
    else:
        resultsFA = FA / Khat
        resultsDE = DE / K

    # print('Missed Detection')
    missed_det = 1 - resultsDE
    print(missed_det)

    # print('False Alarm')
    false_alarm = (resultsFA)
    print(false_alarm)

    return missed_det + false_alarm


if __name__ == "__main__":
    # =========================================== Initialization =========================================== #
    # Number of active users
    K = 16

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

    # print("Number of channel uses::: " + str(nChanlUses))

    # Number of Antennas
    M = 4
    # EbN0dB
    EbN0dB = -1

    # number of pilots (length of the pilots)
    nPilots = 896

    # Channel uses
    nChanlUses = int(nPilots + L * nc / 2)

    # --- Variance of Noise
    sigma2 = 1.0 / ((10 ** (EbN0dB / 10.0)) * B)


    # =========================================== Simulation =========================================== #
    start = time.time()
    print('Number of users: ' + str(K))
    print('Number of antennas: ' + str(M))

    # --- Generate the channel coefficients
    H = (1 / np.sqrt(2)) * (np.random.normal(0, 1, (K, M)) + 1j * np.random.normal(0, 1, (K, M)))

    # --- Generate K msgs at once
    msgs = np.random.randint(0, 2, (K, B))

    Error, Y_msgPart, A = uplink_round(K, nPilots, B, Bf, L, nc, nL, M, sigma2, H, nChanlUses)

    if Error >= 0.05:
        Error = uplink_pilot_round(K, int(nPilots), B, Bf, L, nc, nL, M, sigma2, H, int(nChanlUses), Y_msgPart, A, msgs)
