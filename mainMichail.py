# -*- coding: utf-8 -*-
import numpy as np
from Fasura import FASURA
from PolarCode import PolarCode
from utility import bin2dec, dec2bin, crcEncoder, crcDecoder
import matplotlib.pyplot as plt
import sys

# import random

# random.seed(10)
# No of messages/users

# Message length

B = 32

# Code length

nc = 512

# List decoder

nL = 64

# EbN0dB

EbN0dB = -12.05

# --- Variance of Noise

sigma2 = nc / ((10 ** (EbN0dB / 10.0)) * B)
# print("Variance",sigma2)

# Generate K msgs

msgs = np.random.randint(0, 2, B)

# ================== Add CRC =====================

divisor = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1], dtype=int)  # Choose the divisor
lCRC = len(divisor)  # Number of CRC bits
msgLen = B + lCRC  # Length of the input to the encoder

# =========== Augment CRC remainder to messages ==============
msgCRC = crcEncoder(msgs, divisor)


# Create a polar code object
# ========== Encode the messages ===========
frozenvalues = np.round(np.random.randint(0, 2,nc - msgLen))
polar = PolarCode(nc, msgLen, 1)
codeword, _ = polar.encoder(msgCRC, frozenvalues, -1)





# ========== BPSK modulation ===============

codeword = 2 * codeword - 1

msgCRCHat, PML = polar.listDecoder(codeword, frozenvalues, nL)
thres, flag = np.Inf, -1
isDecoded = 0

    # --- Check the CRC constraint for all message in the list
for l in range(nL):
    check = crcDecoder(msgCRCHat[l, :], divisor)

    if check:
        print("crc successful")
        # --- Check if its PML is larger than the current PML

        if PML[l] < thres:
            flag = l
            thres = PML[l]
            isDecoded = 1
if isDecoded:
    print( sum((msgCRCHat[flag,0:B] + msgs) % 2))
else:
    print("Error\n")
print()