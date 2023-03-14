import numpy as np
import matplotlib.pyplot as plt
import statistics

Detections = np.loadtxt('Detections.out')
Packet_energy = np.loadtxt('Packet_energy.out')
Tot_energy = np.loadtxt('Full_packet_energy.out')

# CDF OF DETECTIONS

# detected_users = np.zeros(Detections.shape[1])
#
# for i in range(Detections.shape[0]):
#     for j in range(Detections.shape[1]-1):
#         if (Detections[i, j] != 0):
#             temp = Detections[i, j]
#             if (Detections[i, j+1] == 0):
#                 Detections[i, j+1] = temp
#
# for i in range(Detections.shape[1]):
#     for j in range(Detections.shape[0]):
#         if int(Detections[j, i]) >= 95:
#             detected_users[i] += 1
#
# detected_users /= Detections.shape[0]
# x = np.arange(17)
# plt.plot(x, detected_users, '--*b', linewidth=1)
#
# plt.xlabel('HARQ rounds')
# plt.ylabel('Prob of 95% detection guarantee')
# plt.title('CDF-detections vs HARQ rounds')
# plt.xticks(x)
# plt.show()





# AVG ENERGY

# user_energy = np.zeros((Packet_energy.shape[1]))
# temp = 0
# for i in range(Packet_energy.shape[1]):
#     user_energy[i] = temp + statistics.mean(Packet_energy[:, i])
#     temp = user_energy[i]
#
# x = np.arange(17)
#
# plt.plot(x, (user_energy/1000), '--*b', linewidth=1)
# plt.hlines(y=(statistics.mean(Tot_energy))/1000, xmin=0, xmax=16, linewidth=2, colors='red')
#
# plt.xlabel('HARQ rounds')
# plt.ylabel('Tx energy spent per round')
# plt.title('')
# plt.xticks(x)
# plt.show()

# AVG DETECTIONS

# detected_users = np.zeros(Detections.shape[1])
#
# # Pre-processing for detections
# for i in range(Detections.shape[0]):
#     for j in range(Detections.shape[1]-1):
#         if (Detections[i, j] != 0):
#             temp = Detections[i, j]
#             if (Detections[i, j+1] == 0):
#                 Detections[i, j+1] = temp
#
#
# for i in range(Detections.shape[1]):
#     detected_users[i] = statistics.mean(Detections[:, i])
#
# x = np.arange(17)
#
# plt.plot(x, detected_users, '-*g', linewidth=1)
#
# plt.xlabel('HARQ rounds')
# plt.ylabel('Avg detections')
# plt.title('')
# # plt.legend(['K = 100, M = 50, Eb_NodB = -10'])
# plt.xticks(x)
# plt.show()
