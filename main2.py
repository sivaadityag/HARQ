# -*- coding: utf-8 -*-
import numpy as np
from PolarCode2 import PolarCode
from utility import crcEncoder, crcDecoder
import matplotlib.pyplot as plt
import sys

# import random

# random.seed(10)

# Codeword length

N=1024


def polar_dec_performance(EbN0dB):
    # No of users

    K = 50

    # Message length

    B = 32

    # Code length

    nc = 1024

    # List decoder

    nL = 64

    # EbN0dB



    # --- Variance of Noise

    sigma2 = nc / ((10 ** (EbN0dB / 10.0)) * B)
    # print("Variance",sigma2)

    # Generate K msgs

    msgs = np.random.randint(0, 2, (K, B))

    # ================== Add CRC =====================

    divisor = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1], dtype=int)  # Choose the divisor
    lCRC = len(divisor)  # Number of CRC bits
    msgLen = B + lCRC  # Length of the input to the encoder

    # =========== Augment CRC remainder to messages ==============

    msgCRC = np.zeros((K,msgLen))  # Initialize an empty list
    for k in range(K):
        msgCRC[k,:]=crcEncoder(msgs[k, :], divisor)
    #print(msgCRC.shape)

    # =========== Create a polar code object

    # frozen values
    frozenvalues = np.round(np.random.randint(low=0, high=2, size=(K, nc - msgLen)))
    #frozenvalues=np.zeros((K, nc - msgLen))
    polar = PolarCode(nc, msgLen, K)

    # ========== Encode the messages ===========

    c_word = np.zeros((K, nc))
    for k in range(K):
        c_word[k, :], _ = polar.encoder(msgCRC[k, :], frozenvalues[k, :], -1)

    # ========== BPSK modulation ===============

    c_word = 2 * c_word - 1

    # ========== Transmission ===============

    r_vec = np.zeros((K, nc))
    for k in range(K):
        r_vec[k, :] = c_word[k, :]  + np.random.normal(0, sigma2, nc)


    # #=========== Checking ARQ =======
    # pos=np.random.randint(0,512,256)
    # for k in range(K):
    #     r_vec[k,448:512]=0

    # =========== Polar decoder function: (Using list decoder) ============

    def polarDecoder(y, frozen, B):

        # ============ Polar decoder ============ #

        msgCRCHat, PML = polar.listDecoder(y, frozen, nL)

        # ============ Check CRC ============ #

        # --- Initialization

        thres, flag = np.Inf, -1
        isDecoded = 0

        # --- Check the CRC constraint for all message in the list

        for l in range(nL):
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
        return isDecoded, msgCRCHat[flag, 0:B]


    # ======= Run the decoding ==========

    recov_msgs = np.zeros((K, B))
    #err = 0
    de = 0

    for k in range(K):
        success, recov_msgs[k, :] = polarDecoder(r_vec[k, :], frozenvalues[k], B)
        if success == 1:
            if sum((recov_msgs[k, :] + msgs[k, :]) % 2) == 0:
                de += 1
            # de += 1

    # for k1 in range(K):
    #     for k2 in range(K):
    #         if sum((recov_msgs[k1, :] + msgs[k2, :]) % 2) == 0:
    #             de += 1
    # print("K:::",K,"de:::",de)
    return ((K-de)/K)

def polar_dec_performanceARQ(EbN0dB,n1,n2):
    
    # No of users

    K = 100

    # Message length

    B = 32

    # Code length

    nc = 1024

    # List decoder

    nL = 64

    # EbN0dB



    # --- Variance of Noise

    sigma2 = nc / ((10 ** (EbN0dB / 10.0)) * B)
    # print("Variance",sigma2)

    # Generate K msgs

    msgs = np.random.randint(0, 2, (K, B))

    # ================== Add CRC =====================

    divisor = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1], dtype=int)  # Choose the divisor
    lCRC = len(divisor)  # Number of CRC bits
    msgLen = B + lCRC  # Length of the input to the encoder

    # =========== Augment CRC remainder to messages ==============

    msgCRC = np.zeros((K,msgLen))  # Initialize an empty list
    for k in range(K):
        msgCRC[k,:]=crcEncoder(msgs[k, :], divisor)
    #print(msgCRC.shape)

    # =========== Create a polar code object

    # frozen values
    frozenvalues = np.round(np.random.randint(low=0, high=2, size=(K, nc - msgLen)))
    #frozenvalues=np.zeros((K, nc - msgLen))
    polar = PolarCode(nc, msgLen, K)

    # ========== Encode the messages ===========

    c_word = np.zeros((K, nc))
    for k in range(K):
        c_word[k, :], _ = polar.encoder(msgCRC[k, :], frozenvalues[k, :], -1)

    # ========== BPSK modulation ===============

    c_word = 2 * c_word - 1

    # ========== Transmission ===============

    r_vec = np.zeros((K, nc))
    for k in range(K):
        r_vec[k, :] = c_word[k, :]  + np.random.normal(0, sigma2, nc)

    
    # #=========== Checking ARQ =======
    r_vec[:,n1::] = 0
    for k in range(K):
        r_vec[k,n1:n2]=0

    # =========== Polar decoder function: (Using list decoder) ============

    def polarDecoder(y, frozen, B):

        # ============ Polar decoder ============ #

        msgCRCHat, PML = polar.listDecoder(y, frozen, nL)

        # ============ Check CRC ============ #

        # --- Initialization

        thres, flag = np.Inf, -1
        isDecoded = 0

        # --- Check the CRC constraint for all message in the list

        for l in range(nL):
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
        return isDecoded, msgCRCHat[flag, 0:B]


    # ======= Run the decoding ==========

    recov_msgs = np.zeros((K, B))
    #err = 0
    de = 0

    for k in range(K):
        success, recov_msgs[k, :] = polarDecoder(r_vec[k, :], frozenvalues[k], B)
        if success == 1:
            de += 1

    # for k1 in range(K):
    #     for k2 in range(K):
    #         if sum((recov_msgs[k1, :] + msgs[k2, :]) % 2) == 0:
    #             de += 1
    # print("K:::",K,"de:::",de)
    return ((K-de)/K)

EbNo_points=4
EbNo_db=np.linspace(6,10,EbNo_points)

err1=np.zeros(EbNo_points)
err2=np.zeros(EbNo_points)
err3=np.zeros(EbNo_points)
err4=np.zeros(EbNo_points)
err5=np.zeros(EbNo_points)

# err6=np.zeros(EbNo_points)
# err7=np.zeros(EbNo_points)

for i in range(EbNo_points):
    err1[i]=polar_dec_performance(EbNo_db[i])
    err2[i]=polar_dec_performanceARQ(EbNo_db[i],(N-64),N)
    err3[i]=polar_dec_performanceARQ(EbNo_db[i],(N-128),N)
    err4[i]=polar_dec_performanceARQ(EbNo_db[i],(N-256),N)
    err5[i]=polar_dec_performanceARQ(EbNo_db[i],(N-512),N)
    # err5[i]=polar_dec_performanceARQ(EbNo_db[i],n1=512-64,n2=512)
    # err6[i]=polar_dec_performanceARQ(EbNo_db[i],n1=512-80,n2=512)
    # err7[i]=polar_dec_performanceARQ(EbNo_db[i],n1=512-96,n2=512)

plt.semilogy(EbNo_db,err1,"-b",label="Full length codeword")
plt.semilogy(EbNo_db,err2,"-m",label="(punctured): 64 bits")
plt.semilogy(EbNo_db,err3,"-c",label="(punctured): 128 bits")
plt.semilogy(EbNo_db,err4,"-g",label="(punctured): 256 bits")
plt.semilogy(EbNo_db,err5,"-y",label="(truncated): 512 bits")
# plt.plot(EbNo_db,err6,"--y",label="ARQ (truncated): 80 bits")
# plt.plot(EbNo_db,err7,"--k",label="ARQ (truncated): 96 bits")

plt.xlabel("EbNo_db")
plt.ylabel("Block error rate")
plt.legend(loc="lower left")
plt.title("(32,1024) code")
plt.show()

# import pathlib
# import os
# current_dir = pathlib.Path(__file__).parent
# #current_dir = "C:\Siva"
# plt.savefig(os.path.join(current_dir, "fig.png"))