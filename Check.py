import numpy as np
Q1 = np.flip(np.genfromtxt("Q" + str(512) + ".csv", delimiter=','))-1
Q2= np.flip(np.genfromtxt("Q" + str(1024) + ".csv", delimiter=','))-1

QQ=Q2[Q2<512]
print(QQ[0],Q1[0])
#print(len(Q),len(Q1),Q[0],Q1[0])


# Q2 = np.flip(np.genfromtxt("Q" + str(512) + ".csv", delimiter=','))-1
# n=512
# j=0
# QQ=np.zeros(n)
# for i in range(1024):
#     if Q[i]<n:
#         QQ[j]=Q[i]
#         j+=1

# QQ = QQ[QQ < n].astype(int)
# Q2 = Q2[Q2 < n].astype(int)
# print(max(QQ),max(Q2),len(QQ),len(Q2),Q2[0],QQ[0])
# #print((np.squeeze(np.where(QQ!=Q2))))
# x=0
# for i in range(n):
#     if QQ[i]!=Q2[i]:
#         x+=1
#         print("Hereee",i)
#         break

# print(i,Q2[i],QQ[i])

# frozenPos = QQ[0:n - k]
# #print(frozenPos[4])
# #print("Hello2",len(frozenPos))
# msgPos = QQ[n - k:n]
#print("Length of msg:::",len(msgPos))