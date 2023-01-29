import numpy as np
import matplotlib.pyplot as plt
#
#
# # P_error = np.array([0.0215, 0.022, 0.02125, 0.036, 0.03875, 0.04625, 0.050283, 0.057316, 0.069166, 0.076383, 0.08828, 0.10375, 0.11364, 0.12200, 0.15352, 0.17565]) * 100
# #
# #
# # Incremental = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240])
# # Incremental = 512 - Incremental
# # plt.plot(Incremental, P_error, '-*b', linewidth=3.0)
# # plt.axhline(y=5, color='r', linestyle='--')
# # plt.xlabel('Codeword_size')
# # plt.ylabel('P_error (%)')
# # plt.title('Varying channel uses')
# # plt.legend(['K = 16, M = 4, Eb_NodB = 1.25'])
# # plt.xticks(Incremental)
# # plt.show()
# #
# #
# rounds = 9
# nIter = 300
# missed_det = np.zeros(rounds)
# false_alarm = np.zeros(rounds)
#
# resultsDE = np.genfromtxt('resultsDE.out')
# resultsFA = np.genfromtxt('resultsFA.out')
#
# # for round in range(rounds):
# #     # print('Missed Detection')
# #     missed_det[round] = sum(1 - resultsDE[:, round]) / float(nIter)
# #     # print(missed_det[round])
# #
# #     # print('False Alarm')
# #     false_alarm[round] = sum(resultsFA[:, round]) / float(nIter)
# #     # print(false_alarm[round])
# #
# #     print("Pe for round:", round , missed_det[round] + false_alarm[round])
# #
# # print("Final simulation result",missed_det + false_alarm)
#
# Total_error = (1-resultsDE) + resultsFA
# # print(Total_error)
# CDF_value = np.zeros(rounds)
# for round in range(rounds):
#
#     for Iter in range(nIter):
#         if Total_error[Iter][round] <= 0.05:
#             CDF_value[round] += 1
# print(CDF_value/300)
#
#
#
#
# import matplotlib.pyplot as plt
#
# nChannelUses = np.array([1352, 1521,  1690, 1859, 2028, 2197, 2366, 2535, 2704])
# plt.plot(nChannelUses,CDF_value/300,'-*b', linewidth=1.5)
# # plt.axhline(y=5, color='r', linestyle='--')
# plt.xlabel('nChannelUses')
# plt.ylabel('Emperical CDF')
# plt.title('Probability that 95% of users are decoded')
# plt.legend(['K = 16, M = 4, Eb_NodB = 1.05'])
# plt.xticks(nChannelUses)
# plt.show()
# from numpy import linalg as LA
# a = np.array([1+1j, 2+2j], dtype = complex)
# b = (abs(a))
# b = b**2
# print(b, LA.norm(a)**2)

Detections = np.loadtxt('Detections.out')
Packet_energy = np.loadtxt('Packet_energy.out')

detected_users = np.zeros(Detections.shape[1])

for i in range(Detections.shape[1]):
    for j in range(Detections.shape[0]):
        if int(Detections[j, i]) >= 95:
            detected_users[i] += 1

detected_users /= Detections.shape[0]
x = np.arange(17)
print(x)
print(detected_users)

import matplotlib.pyplot as plt


plt.plot(x,detected_users,'--or', linewidth=1)
# plt.axhline(y=5, color='r', linestyle='--')
plt.xlabel('HARQ rounds')
plt.ylabel('Emperical CDF')
plt.title('Probability that 95% of users are decoded')
plt.legend(['K = 100, M = 50, Eb_NodB = -10'])
plt.xticks(x)
plt.show()


