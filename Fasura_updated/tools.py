import numpy as np
import matplotlib.pyplot as plt

def plot1Par(x,y,dots):
    if not dots:
        plt.plot(x, y)
    else:
        plt.plot(x, y,marker = 'o')
    plt.grid()
    plt.show()


def aveError(x,xHat):
    nElements = np.size(x)
    error = abs(np.sum(((x - xHat)),axis=None)) ** 2 / nElements
    print('The error is: ' + str(error))
    print()

def checkOverAllof2Sets(S,SHat):
    pass

    
def fromTXtoRX(idxTX,idxRX):
    # --- Find the unique indices
    idxTXUnique = np.unique(idxTX)
    nUsers = len(idxTX)
    map = np.zeros(nUsers)

    for s in idxRX:
        pass


def mseChannel(self, wrongSSIdx, HhatNew, idxSSHat):
    usersRx2userTx = np.array([], dtype=int)
    usersTx2userRx = np.zeros(self.K, dtype=int) - 1
    correctUsersRx = np.array([], dtype=int)
    wrongUserRx = np.array([], dtype=int)

    # --- Initialization
    E = np.zeros((self.K - len(wrongSSIdx) - self.count, self.M), dtype=complex)

    # --- Find the correct order of the channel
    i = 0
    for s, idxHat in enumerate(idxSSHat):
        if idxHat in wrongSSIdx:
            wrongUserRx = np.append(wrongUserRx, s)
            usersRx2userTx = np.append(usersRx2userTx, -1)
            continue
        else:
            pos = np.where(self.idxSS == idxHat)
            pos = np.squeeze(pos)
            if pos.size > 1:
                pos = pos[0]

            E[i, :] = HhatNew[s, :] - self.H[pos, :]
            usersRx2userTx = np.append(usersRx2userTx, pos)
            # usersTx2userRx[pos] = s
            correctUsersRx = np.append(correctUsersRx, s)
            i += 1

    # --- Compute MSE
    Q = np.dot(E.conj().T, E)
    MSE = np.trace(Q) / ((self.K - len(wrongSSIdx) - self.count) * self.M)

    print('\nMean Square Error: {0:.4f}'.format(MSE))

    return usersRx2userTx, usersTx2userRx, correctUsersRx, wrongUserRx, MSE


def checkAccSs(self):
    count = self.checkSS[0] + 1
    FA, DE = 0, 0
    wrongIdx = []
    correctIdx = []
    trueTable = np.ones(self.idxSSHat.size)
    i = 0
    for s in self.idxSSHat:
        # if s in self.supportHat:
        #     continue
        if s in np.unique(self.idxSS):
            self.checkSS[count] = s
            correctIdx.append(s)
            DE += 1
        else:
            FA += 1
            wrongIdx.append(s)
            count += 1
            trueTable[i] = 0
        i += 1
    self.checkSS[0] = count - 1

    print('---------- Energy Detector Results ----------')
    print('Number of Detection: ' + str(DE))
    print('Number of False Alarms: ' + str(FA))
    print("Unique Spre-Seq = " + str(len(np.unique(self.idxSS))))
    return FA, DE, wrongIdx, correctIdx, trueTable