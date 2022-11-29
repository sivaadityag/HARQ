import numpy as np
from PolarCode import PolarCode
from utility import bin2dec, dec2bin, crcEncoder, crcDecoder

K = 10000
B = 100
nc = 512
nL = 64
divisor = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1], dtype=int)
msgLen = B + len(divisor)
CRCcheck = np.zeros(K)

# Create a polar Code object
polar = PolarCode(nc, msgLen, K)


frozenValues = np.round(np.random.randint(low=0, high=2, size=(nc - msgLen, K)))

mis_match = 0


for k in range(K):

    msg = np.random.randint(0, 2, B)


    # --- Add CRC
    msgCRC = crcEncoder(msg, divisor)

    # --- polar encoder
    codeword, _ = polar.encoder(msgCRC, frozenValues[:, k], -1)

    # Modulate over AWGN

    SNR_dB = 0
    SNR = 10 ** (SNR_dB / 10)
    rate = msgLen/nc

    nvar = 1 / (2 * rate * SNR)

    # ========== BPSK modulation ===============

    codeword = 1 - 2 * codeword
    noise = np.sqrt(nvar) * np.random.randn(nc)

    # Tx via AWGN

    Y = codeword + noise

    # Decoder

    msgCRCHat, PML = polar.listDecoder(codeword, frozenValues[:, k], nL)
    msgCRCHat = msgCRCHat.astype(int)

    # ============ Check CRC ============ #
    # --- Initialization
    thres, flag = np.Inf, -1

    crc_var = 0

    # --- Check the CRC constraint for all message in the list
    for l in range(nL):

        check = crcDecoder(msgCRCHat[l, :], divisor)

        # print(" Checking CRC", l, check)

        if check:
            crc_var += 1

            CRCcheck[k] = crc_var

            # --- Check if its PML is larger than the current PML
            if PML[l] < thres:
                flag = l
                thres = PML[l]


    if thres == np.Inf:
        # --- Return the message with the minimum PML
        # print(PML)
        flag = np.argmin(PML)

    # print("Final list decoding path", flag)

    Decoded_msg = msgCRCHat[flag, :]


    for i in range(msgLen):
        if msgCRC[i] != Decoded_msg[i]:
            mis_match += 1

# print("Checking CRC", CRCcheck)
print("Error %",mis_match/(nc*K) )