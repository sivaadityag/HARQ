# -*- coding: utf-8 -*-
import numpy as np
from PolarCode2 import PolarCode
from utility import crcEncoder, crcDecoder
import matplotlib.pyplot as plt
import sys

class Full_Pdecoder():
    def __init__(self,N,K,B,nL):
        self.nc=N
        self.K=K
        self.B=B
        self.nL=nL
        

    def polar_dec_performance(self,EbN0dB):
        
        
        # --- Variance of Noise

        sigma2 = self.nc / ((10 ** (EbN0dB / 10.0)) * self.B)
        # print("Variance",sigma2)

        # Generate K msgs

        msgs = np.random.randint(0, 2, (self.K, self.B))

        # ================== Add CRC =====================

        divisor = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1], dtype=int)  # Choose the divisor
        lCRC = len(divisor)  # Number of CRC bits
        msgLen = self.B + lCRC  # Length of the input to the encoder

        # =========== Augment CRC remainder to messages ==============

        msgCRC = np.zeros((self.K,msgLen))  # Initialize an empty list
        for k in range(self.K):
            msgCRC[k,:]=crcEncoder(msgs[k, :], divisor)
        #print(msgCRC.shape)

        # =========== Create a polar code object

        # frozen values
        frozenvalues = np.round(np.random.randint(low=0, high=2, size=(self.K, self.nc - msgLen)))
        #frozenvalues=np.zeros((self.K, nc - msgLen))
        polar = PolarCode(self.nc, msgLen, self.K)

        # ========== Encode the messages ===========

        c_word = np.zeros((self.K, self.nc))
        for k in range(self.K):
            c_word[k, :], _ = polar.encoder(msgCRC[k, :], frozenvalues[k, :], -1)

        # ========== BPSK modulation ===============

        c_word = 2 * c_word - 1

        # ========== Transmission ===============

        r_vec = np.zeros((self.K, self.nc))
        for k in range(self.K):
            r_vec[k, :] = c_word[k, :]  + np.random.normal(0, sigma2, self.nc)

        
        # #=========== Checking ARQ =======
        # r_vec[:,n1::] = 0
        # for k in range(self.K):
        #     r_vec[k,n1:n2]=0

        # =========== Polar decoder function: (Using list decoder) ============

        def polarDecoder(y, frozen):

            # ============ Polar decoder ============ #

            msgCRCHat, PML = polar.listDecoder(y, frozen, self.nL)

            # ============ Check CRC ============ #

            # --- Initialization

            thres, flag = np.Inf, -1
            isDecoded = 0

            # --- Check the CRC constraint for all message in the list

            for l in range(self.nL):
                check = crcDecoder(msgCRCHat[l, :], divisor)

                if check:
                    # print("crc successful")
                    # --- Check if its PML is larger than the current PML

                    if PML[l] < thres:
                        flag = l
                        thres = PML[l]
                        isDecoded = 1

            # if thres == np.Inf:
            #     # --- Return the message with the minimum PML
            #     flag = np.squeeze(np.where(PML == PML.min()))
                #flag=flag.min()
            return isDecoded, msgCRCHat[flag, 0:self.B]


        # ======= Run the decoding ==========

        recov_msgs = np.zeros((self.K, self.B))
        #err = 0
        de = 0

        for k in range(self.K):
            success, recov_msgs[k, :] = polarDecoder(r_vec[k, :], frozenvalues[k])
            if success == 1:
                de += 1

        # for k1 in range(K):
        #     for k2 in range(K):
        #         if sum((recov_msgs[k1, :] + msgs[k2, :]) % 2) == 0:
        #             de += 1
        # print("K:::",K,"de:::",de)
        return ((self.K-de)/self.K)