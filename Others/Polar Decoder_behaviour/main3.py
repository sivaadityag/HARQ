import numpy as np
from Try.Punctured_Pdecoder import Punctured_Pdecoder
from Try.Full_Pdecoder import Full_Pdecoder
import matplotlib.pyplot as plt

N=1024
B=32
nL=64
K=100

Full_code=Full_Pdecoder(N,K,B,nL)
Punc_code=Punctured_Pdecoder(N,K,B,nL)


EbNo_db=np.arange(6,12,0.25)

err1=np.zeros(len(EbNo_db))
err2=np.zeros(len(EbNo_db))
err3=np.zeros(len(EbNo_db))
err4=np.zeros(len(EbNo_db))
err5=np.zeros(len(EbNo_db))

niter=20
err1_fixed=0
err2_fixed=0
err3_fixed=0
err4_fixed=0
err5_fixed=0

for i in range(len(EbNo_db)):
    for iter in range(niter):
        err1_fixed+=Full_code.polar_dec_performance(EbNo_db[i])
        err2_fixed+=Punc_code.polar_dec_performanceARQ(EbNo_db[i],(N-64),N)
        err3_fixed+=Punc_code.polar_dec_performanceARQ(EbNo_db[i],(N-128),N)
        err4_fixed+=Punc_code.polar_dec_performanceARQ(EbNo_db[i],(N-256),N)
        err5_fixed+=Punc_code.polar_dec_performanceARQ(EbNo_db[i],(N-512),N)
    err1[i]=err1_fixed/niter
    err2[i]=err2_fixed/niter
    err3[i]=err3_fixed/niter
    err4[i]=err4_fixed/niter
    err5[i]=err5_fixed/niter
    # err5[i]=polar_dec_performanceARQ(EbNo_db[i],n1=512-64,n2=512)
    # err6[i]=polar_dec_performanceARQ(EbNo_db[i],n1=512-80,n2=512)
    # err7[i]=polar_dec_performanceARQ(EbNo_db[i],n1=512-96,n2=512)

plt.semilogy(EbNo_db,err1,"-b",label="Full length codeword")
plt.semilogy(EbNo_db,err2,"-m",label="(punctured): 64 bits")
plt.semilogy(EbNo_db,err3,"-c",label="(punctured): 128 bits")
plt.semilogy(EbNo_db,err4,"-g",label="(punctured): 256 bits")
plt.semilogy(EbNo_db,err5,"-r",label="(punctured): 512 bits")
# plt.plot(EbNo_db,err6,"--y",label="ARQ (truncated): 80 bits")
# plt.plot(EbNo_db,err7,"--k",label="ARQ (truncated): 96 bits")

# plt.xlabel("EbNo_db")
# plt.ylabel("Block error rate")
# plt.legend(loc="lower left")
# plt.title("(32,1024) code")
# plt.show()

import pathlib
import os
current_dir = pathlib.Path(__file__).parent
plt.savefig(os.path.join(current_dir, "fig.png"))