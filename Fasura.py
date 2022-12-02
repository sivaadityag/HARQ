# -*- coding: utf-8 -*-
import numpy as np
from PolarCode import PolarCode
from utility import bin2dec, dec2bin, crcEncoder, crcDecoder


class FASURA:
    def __init__(self, K, n_Pilots, B, Bf, L, nc, nL, M, sigma2, H, P, A, nChanl_uses):
        self.K = K
        self.Bf = Bf  # number of bits of the first part of the message
        self.Bs = B - Bf  # number of bits of the second part of the message
        self.L = L  # Length of spreading sequence
        self.J = 2 ** Bf  # Number of spreading sequence
        self.nc = nc  # length of code
        self.nL = nL  # List size
        # self.nChanlUses = int((nc / np.log2(4)) * L + nPilots)
        self.nChanlUses = nChanl_uses
        self.nDataSlots = int(nc / np.log2(4))
        self.M = M  # Number of antennas
        self.nPilots = n_Pilots  # number of pilot symbols

        # self.S = (1 - 2 * np.round(np.random.randint(low=0, high=2, size=(self.nChanlUses, self.J)))) + 1j * (
        #         1 - 2 * np.round(np.random.randint(low=0, high=2, size=(self.nChanlUses, self.J))))

        # Pilots
        # self.P = self.S[0:self.nPilots, :] / np.sqrt(2.0 * self.nChanlUses)
        self.P = P

        # spreading sequence master set
        # self.A = self.S[self.nPilots::, :] / np.sqrt(4.0 * self.nChanlUses)
        self.A = A

        # Polynomial for CRC coding
        if K <= 300:
            self.divisor = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1], dtype=int)
        else:
            self.divisor = np.array([1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1], dtype=int)

        self.lCRC = len(self.divisor)  # Number of CRC bits
        self.msgLen = self.Bs + self.lCRC  # Length of the input to the encoder
        self.frozenValues = np.round(np.random.randint(low=0, high=2, size=(nc - self.msgLen, self.J)))

        # Create a polar Code object
        self.polar = PolarCode(nc, self.msgLen, K)

        self.sigma2 = sigma2  # variance of the noise
        self.interleaver = np.zeros((self.nc, self.J), dtype=int)
        for j in range(self.J):
            self.interleaver[:, j] = np.random.choice(self.nc, self.nc, replace=False)

        self.msgs = np.zeros((K, Bf + self.Bs))  # Store the active messages
        self.msgsHat = np.zeros((K, Bf + self.Bs))  # Store the recovered messages
        # self.count = 0  # Count the number of recovered msgs in this round
        self.Y = np.zeros((self.nChanlUses, M))
        self.Y_contrib = np.zeros((self.nChanlUses, M))
        # self.idxSSDec = np.array([], dtype=int)
        # self.idxSSHat = np.array([], dtype=int)  # To store the new recovered sequences
        self.symbolsHat = np.zeros((self.K, self.nDataSlots), dtype=complex)
        self.NOPICE = 0
        # self.check = 0  # Stopping condition when two subsequent SIC rounds have no new detections!
        self.flag = 0 # HARQ variable

    def transmitter(self, msgBin, H):

        """
        Function to encode the messages of the users
        Inputs: 1. the message of the users in the binary form, dimensions of msgBin, K x B
                2. Channel
        Output: The sum of the channel output before noise, dimensions of Y, n x M
        """

        # ===================== Initialization ===================== #
        Y = np.zeros((self.nChanlUses, self.M), dtype=complex)

        # --- For all active users
        for k in range(self.K):

            # --- Save the active message k
            self.msgs[k, :] = msgBin[k, :]

            # --- Break the message into to 2 parts
            # First part, Second part
            mf = msgBin[k, 0:self.Bf]
            ms = msgBin[k, self.Bf::]

            # --- Find index of the spreading sequence
            idxSS = bin2dec(mf)
            # print(idxSS)

            # --- Add CRC
            msgCRC = crcEncoder(ms, self.divisor)

            # --- polar encoder
            codeword, _ = self.polar.encoder(msgCRC, self.frozenValues[:, idxSS], k)

            # --- Interleaver
            codeword = codeword[self.interleaver[:, idxSS]]

            # --- QPSK modulation
            symbols = QPSK(codeword)

            # --- Initialize two temp Matrices
            YTempPilots = np.zeros((self.nPilots, self.M), dtype=complex)
            YTempSymbols = np.zeros((self.nDataSlots * self.L, self.M), dtype=complex)

            # --- For Pilots (PH)
            for m in range(self.M):
                YTempPilots[:, m] = self.P[:, idxSS] * H[k, m]

            # --- For Symbols (QH)
            A = np.zeros((self.nDataSlots * self.L), dtype=complex)
            for t in range(self.nDataSlots):
                A[t * self.L: (t + 1) * self.L] = self.A[t * self.L: (t + 1) * self.L, idxSS] * symbols[t]

            for m in range(self.M):
                YTempSymbols[:, m] = A * H[k, m]

            # --- Add the new matrix to the output signal
            Y += np.vstack((YTempPilots, YTempSymbols))

        return Y


    def pilot_transmitter(self, msgBin , H):

        Y_pilots = np.zeros((self.nPilots, self.M), dtype=complex)
        # --- For all active users
        for k in range(self.K):

            # --- Save the active message k
            self.msgs[k, :] = msgBin[k, :]

            # --- Break the message into to 2 parts
            # First part, Second part
            mf = msgBin[k, 0:self.Bf]

            # --- Find index of the spreading sequence
            idxSS = bin2dec(mf)

            # --- Initialize two temp Matrices
            YTempPilots = np.zeros((self.nPilots, self.M), dtype=complex)

            # --- For Pilots (PH)
            for m in range(self.M):
                YTempPilots[:, m] += self.P[:, idxSS] * H[k, m]

            #Additive channel

            Y_pilots += Y_pilots + YTempPilots
        return Y_pilots

    def receiver(self, Y, decoded_users, harq_factor, prev_msgs):

        """
        Function to recover the messages of the users from noisy observations
        Input:  The received signal, dimensions of Y, n x M
        Output: Probability of Detection and False Alarm
        """

        # --- Save the received signal
        self.Y = Y.copy()
        self.count = 0  # Count the number of recovered msgs in this round
        early_detec = decoded_users # HARQ parameter
        self.idxSSDec = np.array([], dtype=int)
        self.idxSSHat = np.array([], dtype=int)
        self.Y_contrib = np.zeros((self.nChanlUses, self.M), dtype=complex)
        self.decoded_symbols = list()

        global H_total

        # =========================================== Receiver  =========================================== #


        # ======================== HARQ Final round exit Condition ======================================== #

        if (self.K - (self.count+early_detec)) == 0:
            print("All rounds of HARQ concluded")
            return 0, 0, 0, np.zeros((self.nChanlUses, self.M), dtype=complex), 0, np.zeros((self.nChanlUses, self.M), dtype=complex), 1, 0

        # ======================== Pilot / Spreading Sequence Detector ==================================== #

        self.idxSSHat = energyDetector(self, self.Y, self.K - (self.count+early_detec))

        # print("Checking Energy detector Rx sequences", self.idxSSHat.shape, self.idxSSHat)

        # ======================== Channel estimation (Pilots) ============================================ #

        HhatNew = channelEst(self)

        # ======================== Symbol estimation and Polar Code ======================== #

        userDecRx, notUserDecRx, symbolsHatHard, msgsHat2Part, userDecRx_symbols = decoder(self, HhatNew, self.idxSSHat)

        # --- Add the decoded indices
        self.idxSSDec = np.append(self.idxSSDec, self.idxSSHat[userDecRx])

        # --- HARQ tweak

        for sym in range(len(userDecRx_symbols)):
            self.decoded_symbols.append(np.array(userDecRx_symbols[sym]))

        # print("Checking Decoded Rx sequences", self.idxSSDec)

        # ======================== Exit Condition ======================== #

        # --- No new decoded user

        if userDecRx.size == 0:

            print('Present HARQ round failed: Insufficient information')
            return 0, 0, 0, np.zeros((self.nChanlUses, self.M), dtype=complex), 0, np.zeros((self.nChanlUses, self.M), dtype=complex), 0, 0

        # ======================== Channel estimation (P + Q) ======================== #

        # --- Estimate the channel of the correct users

        # Use the received signal

        self.Y = Y.copy()

        HhatNewDec, H_total = channelEstWithDecUsers(self, Y, self.idxSSDec, symbolsHatHard[userDecRx], harq_factor) # Using original recieved value of Y for estimating H

        # ================================== SIC ================================== #

        # Only one user is decoded

        if userDecRx.size == 1:

            # Only one user left
            if msgsHat2Part.shape[0] == 1:
                if not isIncluded(self, msgsHat2Part, self.idxSSHat[userDecRx]):
                    Hsub = np.squeeze(HhatNewDec.reshape(self.M, 1))
                    subInter(self, np.squeeze(symbolsHatHard), self.idxSSHat, Hsub, harq_factor)
                    saveUser(self, msgsHat2Part, self.idxSSHat[userDecRx])

            # More than one user left
            else:
                if not isIncluded(self, msgsHat2Part[userDecRx, :], self.idxSSHat[userDecRx]):
                    Hsub = HhatNewDec
                    subInter(self, np.squeeze(symbolsHatHard[userDecRx, :]), self.idxSSHat[userDecRx], Hsub, harq_factor)
                    saveUser(self, msgsHat2Part[userDecRx, :], self.idxSSHat[userDecRx])

        # More than one user decode

        else:

            Hsub = HhatNewDec

            for g in range(userDecRx.size):
                if not isIncluded(self, msgsHat2Part[userDecRx[g], :], self.idxSSHat[userDecRx[g]]):
                    subInter(self, symbolsHatHard[userDecRx[g]], self.idxSSHat[userDecRx[g]], Hsub[g, :], harq_factor)
                    saveUser(self, msgsHat2Part[userDecRx[g], :], self.idxSSHat[userDecRx[g]])

        # ======================== Find the performance ======================== #

        # de, fa = checkPerformance(self)
        # print('Number of Detections: ' + str(de))
        # print('Number of False Alarms: ' + str(fa))

        # ======================== Exit Condition ======================== #

        # Feedback for the next round

        DE, FA, detected_msgs, Y_DE = checkPerformance(self, np.array(self.decoded_symbols), self.idxSSDec, H_total, prev_msgs)
        return DE, FA, self.count, Y_DE, np.array(detected_msgs), self.Y_contrib, 0, 1


# ============================================ Functions ============================================ #
# === General Functions
def QPSK(data):
    # Reshape data
    data = np.reshape(data, (2, -1))
    symbols = 1.0 - 2.0 * data

    return symbols[0, :] + 1j * symbols[1, :]


def LMMSE(y, A, Qx, Qn):

    """
        Function that implements the LMMSE Traditional and Compact Form estimator
        for the system y = Ax + n

        Input: y: received vector, A: measurement matrix, Qx: covariance of x, Qn: covariance of the noise
        Output: xHat: estimate of x, MSE: covariance matrix of the error
    """

    r, c = A.shape

    # ========================= Traditional LMMSE ========================= #
    # https://en.wikipedia.org/wiki/Minimum_mean_square_error
    if r <= c:

        # Covariance Cov(X,Y)
        Qxy = np.dot(Qx, A.conj().T)

        # Covariance of Y
        Qy = np.dot(np.dot(A, Qx), A.conj().T) + Qn

        # Inverse of Covariance of Y
        QyInv = np.linalg.inv(Qy)

        # --- Filter
        F = np.dot(Qxy, QyInv)


    # ========================= Compact LMMSE ========================= #
    else:
        # --- Inverse of matrices
        QxInv = np.linalg.inv(Qx)

        QnInv = np.linalg.inv(Qn)

        # --- Second Term
        W2 = np.dot(A.conj().T, QnInv)

        # --- First term
        W1 = QxInv + np.dot(W2, A)
        W1Inv = np.linalg.inv(W1)

        # --- Filter
        F = np.dot(W1Inv, W2)

    # --- Estimates
    xHat = np.dot(F, y)

    return xHat


def isIncluded(self, second, idxSS):
    # --- Convert the decimal index of the spreading sequence to the binary string of length self.Bf
    first = dec2bin(np.hstack((idxSS, idxSS)), self.Bf)
    # --- Concatenate the two parts
    msgHat = np.append(first[0, :], second)

    # --- Check if we recovered this message
    for i in range(self.count):
        # --- Binary Addition
        binSum = sum((msgHat + self.msgsHat[i, :]) % 2)

        if binSum == 0:
            return 1
    return 0


def subInter(self, symbols, idxSS, h, factor):

    # Define a temp Matrix and fill the matrix

    YTempPilots = np.zeros((self.nPilots, self.M), dtype=complex)
    YTempSymbols = np.zeros((self.nDataSlots * self.L, self.M), dtype=complex)

    # Temporary fix: Need to check LMMSE dimensions when only one user is present

    h = np.reshape(h, (self.M, 1))

    # --- For Pilots

    for m in range(self.M):

        YTempPilots[:, m] = np.squeeze(self.P[:, idxSS]) * h[m]

    # --- For Symbols

    A = np.zeros((self.nDataSlots * self.L), dtype=complex)

    for t in range(self.nDataSlots):
        A[t * self.L: (t + 1) * self.L] = np.squeeze(self.A[t * self.L: (t + 1) * self.L, idxSS]) * symbols[t]

    for m in range(self.M):
        YTempSymbols[:, m] = A * h[m]

    # Subtract (SIC)

    # self.Y -= np.vstack((YTempPilots, YTempSymbols))
    # self.Y[factor::, :] = 0 # HARQ variable for varying channel uses and appropriate adjustment for SIC
    self.Y_contrib += np.vstack((YTempPilots, YTempSymbols))

def saveUser(self, msg2Part, idxSS):

    self.msgsHat[self.count, :] = np.concatenate(
        (np.squeeze(dec2bin(np.array([idxSS]), self.Bf)), np.squeeze(msg2Part)), 0)
    self.count += 1


def checkPerformance(self, symbols, idxSS, H, old_msgs):
    numDE, numFA, detections = 0, 0, list()
    Y_detected = np.zeros((self.nChanlUses, self.M), dtype=complex)

    for i in range(self.count):
        flag = 0
        for k in range(self.K):
            binSum = sum((self.msgs[k, :] + self.msgsHat[i, :]) % 2)

            if old_msgs is not None and binSum == 0:
                for l in range(old_msgs.shape[0]):
                    binSum2 = sum((self.msgs[k, :] + old_msgs[l, :]) % 2)

                    if binSum2 == 0:
                        binSum = 1

            if binSum == 0:
                flag = 1
                break
        if flag == 1:

            numDE += 1

            # --- HARQ changes

            # Define a temp Matrix and fill the matrix

            YTempPilots = np.zeros((self.nPilots, self.M), dtype=complex)
            YTempSymbols = np.zeros((self.nDataSlots * self.L, self.M), dtype=complex)

            # Temporary fix: Need to check LMMSE dimensions when only one user is present

            # --- For Pilots

            # Temporary fix of LMMSE

            h = np.reshape(H[i], (self.M, 1))

            for m in range(self.M):
                YTempPilots[:, m] = np.squeeze(self.P[:, idxSS[i]]) * h[m]

            # --- For Symbols

            A = np.zeros((self.nDataSlots * self.L), dtype=complex)

            for t in range(self.nDataSlots):
                A[t * self.L: (t + 1) * self.L] = np.squeeze(self.A[t * self.L: (t + 1) * self.L, idxSS[i]]) * \
                                                  symbols[i][t]

            for m in range(self.M):
                YTempSymbols[:, m] = A * h[m]

            Y_detected += np.vstack((YTempPilots, YTempSymbols))

            detections.append(self.msgsHat[i, :])

        else:
            numFA += 1

    return numDE, numFA, detections, Y_detected


# === Energy Detector
def energyDetector(self, y, K):

    # --- Energy Per Antenna
    energy = np.linalg.norm(np.dot(self.P.conj().T, y[0:self.nPilots, :]), axis=1) ** 2

    pivot = self.nPilots

    for t in range(self.nDataSlots):
        energy += np.linalg.norm(np.dot(self.A[t * self.L: (t + 1) * self.L, :].conj().T, y[pivot + t * self.L: pivot + (t + 1) * self.L, :]), axis=1) ** 2

    return np.argpartition(energy, -K)[-K:]


# ==== Functions For Channel Estimation
def channelEst(self):
    return LMMSE(self.Y[0:self.nPilots, :], self.P[:, self.idxSSHat], np.eye(len(self.idxSSHat)),
                 np.eye(self.nPilots) * self.sigma2)


def channelEstWithErrors(self, symbolsHatHard):
    K = symbolsHatHard.shape[0]

    # -- Create A
    A = np.zeros((self.nChanlUses, K), dtype=complex)
    for k in range(K):
        Atemp = np.zeros((self.nDataSlots * self.L), dtype=complex)
        for t in range(self.nDataSlots):
            Atemp[t * self.L: (t + 1) * self.L] = self.A[t * self.L: (t + 1) * self.L, self.idxSSHat[k]] * \
                                                  symbolsHatHard[k, t]

        A[:, k] = np.hstack((self.P[:, self.idxSSHat[k]], Atemp))

    return LMMSE(self.Y, A, np.eye(K), np.eye(self.nChanlUses) * self.sigma2)


def channelEstWithDecUsers(self, Y, decUsersSS, symbolsHatHard, factor):

    for i in range(self.count - 1, -1, -1):

        symbolsHatHard = np.vstack((self.symbolsHat[i, :], symbolsHatHard))

    # K = Only detected users

    K = decUsersSS.size

    # -- Create A
    A = np.zeros((self.nChanlUses, K), dtype=complex)

    for k in range(K):

        Atemp = np.zeros((self.nDataSlots * self.L), dtype=complex)

        for t in range(self.nDataSlots):
            if K > 1:
                Atemp[t * self.L: (t + 1) * self.L] = self.A[t * self.L: (t + 1) * self.L, decUsersSS[k]] * symbolsHatHard[k, t]
            else:
                Atemp[t * self.L: (t + 1) * self.L] = (self.A[t * self.L: (t + 1) * self.L, decUsersSS[k]] * symbolsHatHard[:,t])
        if K > 1:
            A[:, k] = np.hstack((self.P[:, decUsersSS[k]], Atemp))
        else:
            A[:, k] = np.hstack((self.P[:, decUsersSS[k]], Atemp))

    Hhat = LMMSE(Y, A, np.eye(K), np.eye(self.nChanlUses) * self.sigma2)

    for i in range(self.count):
        subInter(self, self.symbolsHat[i, :], decUsersSS[i], Hhat[i, :], factor)

    return Hhat[self.count::, :], Hhat

# ==== Functions For Symbol Estimation
def symbolsEst(Y, H, A, Qx, Qn, nSlots, L):
    """
        Function that implements the symbols estimation for spread-based MIMO system

        X: diagonal matrix contains the symbol for each user, H: channel matrix, N: noise

        Input: Y: received vector, H: channel matrix,
               A: spreading sequence master set (for all slots, only active) dim(A) = (L x nSlots) x totalNumber of Spreading Sequence,
               Qx: covariance of x, Qn: covariance of the noise
               nSlots: number of slots which is equal to the number of symbols
               L: Length of the spreading sequence
        Output: xHat: estimate of x, MSE: covariance matrix of the error
    """

    K = H.shape[0]

    symbolsHat = np.zeros((K, nSlots), dtype=complex)

    # --- For all Symbols
    for t in range(nSlots):
        symbolsHat[:, t] = symbolEstSubRoutine(Y[t * L: (t + 1) * L, :], H, A[t * L:(t + 1) * L, :], Qx, Qn)

    return symbolsHat


def symbolEstSubRoutine(Y, H, S, Qx, Qn):
    """
        Function that implements the symbol estimation for spread-based MIMO system Y = SXH + N
        where Y: received matrix, S: spreading sequence matrix (only active columns),
        X: diagonal matrix contains the symbol for each user, H: channel matrix, N: noise

        Input: Y: received vector, H: channel matrix, S: spreading sequence matrix ,
               Qx: covariance of x, Qn: covariance of the noise
        Output: xHat: estimate of x, MSE: covariance matrix of the error
    """

    K, M = H.shape  # K: number of users, M: number of antennas
    L = S.shape[0]  # L: length of the spreading sequence

    # --- First Step:
    # Convert the system from Y = SXH + N, to y = Ax + n, where A contains the channel and sequence

    A = np.zeros((L * M, K), dtype=complex)

    for m in range(M):
        if K == 1:
            A[m * L:L * (m + 1), :] = S * H[:, m]
        else:
            # --- Diagonalize H
            A[m * L:L * (m + 1), :] = np.dot(S, np.diag(H[:, m]))

    # --- Second Step:

    # Flat Y
    y = np.ndarray.flatten(Y.T)

    # Estimate the symbols
    return LMMSE(y, A, Qx, Qn)


# ==== Decoder
def decoder(self, H, idxSSHat):
    K = idxSSHat.size

    symbolsHatHard = np.zeros((K, self.nDataSlots), dtype=complex)

    # ==================================== Symbol Estimation Decoder ============================================== #
    symbolsHat = symbolsEst(self.Y[self.nPilots::, :], H, self.A[:, idxSSHat], np.eye(K) * 2,
                            np.eye(self.L * self.M) * self.sigma2, self.nDataSlots, self.L)


    # ==================================== Channel Decoder ============================================== #
    userDecRx = np.array([], dtype=int)
    userDecRx_symbols = list() # HARQ tweak
    notUserDecRx = np.array([], dtype=int)
    msgsHat = np.zeros((K, self.Bs), dtype=int)

    for s in range(symbolsHat.shape[0]):

        # Form the codeword
        cwordHatSoft = np.concatenate((np.real(symbolsHat[s, :]), np.imag(symbolsHat[s, :])), 0)

        # Interleaver
        cwordHatSoftInt = np.zeros(self.nc)
        cwordHatSoftInt[self.interleaver[:, self.idxSSHat[s]]] = cwordHatSoft

        # Call polar decoder
        cwordHatHard, isDecoded, msgHat = polarDecoder(self, cwordHatSoftInt, self.idxSSHat[s])

        # What is this 256 condition?
        # Ans: Strike a balance b/w FA and MD

        if isDecoded == 1 and sum(abs(((cwordHatSoftInt < 0) * 1 - cwordHatHard)) % 2) > 256:
            isDecoded = 0

        # Why is it necessary to store all symbols including those of which that are not decoded?

        symbolsHatHard[s, :] = QPSK(cwordHatHard[self.interleaver[:, self.idxSSHat[s]]])

        msgsHat[s, :] = msgHat

        if isDecoded:
            userDecRx = np.append(userDecRx, s)
            userDecRx_symbols.append(list(symbolsHatHard[s, :]))
        else:
            notUserDecRx = np.append(notUserDecRx, s)

    return userDecRx, notUserDecRx, symbolsHatHard, msgsHat, np.array(userDecRx_symbols)


def polarDecoder(self, bitsHat, idxSSHat):
    # ============ Polar decoder ============ #

    # position=np.random.randint(low=0,high=1024,size=400)
    # bitsHat[800:1024]=0
    msgCRCHat, PML = self.polar.listDecoder(bitsHat, self.frozenValues[:, idxSSHat], self.nL)

    # ============ Check CRC ============ #
    # --- Initialization
    thres, flag = np.Inf, -1
    isDecoded = 0

    # --- Check the CRC constraint for all message in the list
    for l in range(self.nL):
        check = crcDecoder(msgCRCHat[l, :], self.divisor)

        if check:

            # --- Check if its PML is larger than the current PML
            if PML[l] < thres:
                flag = l
                thres = PML[l]
                isDecoded = 1

    if thres == np.Inf:
        # --- Return the message with the minimum PML

        flag = np.argmin(PML)


    # --- Encode the estimated message

    codewordHat, _ = self.polar.encoder(msgCRCHat[flag, :], self.frozenValues[:, idxSSHat], -1)

    return codewordHat, isDecoded, msgCRCHat[flag, 0:self.Bs]
