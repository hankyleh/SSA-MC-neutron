import numpy
import math
import sys
import os
import ctypes
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from mpi4py import MPI
comm = MPI.COMM_WORLD
root = 0
rank = comm.Get_rank()
num_procs = comm.Get_size()


date = datetime.now()
date_str = date.strftime("%Y-%m-%d %H:%M:%S")

###################### initialize inputs ########################
#################################################################


Gens = int(sys.argv[1])                                         #
                                                                #
p_nu = [0, 0, 1]#
nu_s = [0, 0.5, 0.5]                                            #
#nubar = 4.4631                                                  # 
                                                                #
S = 0.000                                                       #
                                                                #
keff = float(sys.argv[3])                                       #
m = 1                                                           #
maxN = int(sys.argv[4])
                                                                #
exename = "./ssaMC.so"                                          #
outputdir = "./out"                                             #
caseName = sys.argv[2]                                          #
caseDir = "/" + caseName                                        #
                                                                #
plotmin = -2                                                    #
plotmax = 4                                                     #
                                                                #
saveData=False                                                  #
                                                                #
#################################################################
#################################################################





### Normalize Nu vector ###
s = sum(p_nu)
tmp = [x / s for x in p_nu]
p_nu = tmp

s = sum(nu_s)
tmp = [x / s for x in nu_s]
nu_s = tmp

nu_file = open("p_nu_f.txt", "w")
for i in p_nu:
    nu_file.write(str(i)+" ")
nu_file = open("q_nu_s.txt", "w")
for i in nu_s:
    nu_file.write(str(i)+" ")

nubar = numpy.dot(p_nu, numpy.linspace(0, len(p_nu)-1, len(p_nu)))   

if rank == root:
    if not os.path.exists(outputdir+caseDir):
        os.mkdir(outputdir+caseDir)

probFiss = keff/nubar
probCap = 1-probFiss
probLeak = 0

gens_per_thread = math.ceil(Gens/num_procs)

path = ctypes.CDLL(exename)

datapath = str.encode(f"out/{caseName}/{caseName}-data-{str(rank)}.txt")
path.xpdf.argtypes = [ctypes.c_int,ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,ctypes.c_double, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
path.xpdf(m, gens_per_thread, probCap, probFiss, probLeak, S, ctypes.c_char_p(datapath), rank, maxN)



fromC = f"out/{caseName}/{caseName}-data-{str(rank)}.txt"
M = numpy.array(numpy.loadtxt(fromC, comments="#", delimiter=",", unpack=False))
rank_max = M.max()
total = num_procs * len(M)
if rank != 0:
    comm.send(rank_max, dest=0, tag=10*rank)
if rank == 0:
    max_list = [rank_max]
    for i in numpy.linspace(1, num_procs-1, num_procs-1):
        tmp=comm.recv(source=i, tag=10*i)
        max_list += [tmp]
    abs_max = max(max_list)

    n=math.ceil(total**(1/2.5))
    logs = numpy.linspace(-6, math.log10(M.max()), n)
    bins = numpy.power(10, logs)
    spaces = (bins[2:-1]-bins[1:-2])
    meshpoints = numpy.array(bins[1:-2] + spaces/2)
    
    for i in numpy.linspace(1, num_procs-1, num_procs-1):
        comm.send(bins, dest=i, tag=11*i)
if rank != root:
    bins=comm.recv(source=0, tag=11*rank)
    spaces = (bins[2:-1]-bins[1:-2])
    meshpoints = numpy.array(bins[1:-2] + spaces/2)

print(f"Begin binning on node {rank}")

counts = []
for i in range(len(meshpoints)):
    temp_p = 0.0
    temp_p = numpy.dot((M<bins[i+1])*1, (bins[i]<M)*1)

    counts += [temp_p]

if rank != root:
    comm.send(counts, dest=0, tag=12*rank)
if rank == root:
    counts = numpy.array(counts)

    for i in numpy.linspace(1, num_procs-1, num_procs-1):
        counts += comm.recv(source=i, tag=12*i)

    proportion = counts
    proportion = numpy.array([counts[i] / (total*spaces[i]) for i in range(len(counts))])
    sigma = numpy.array([math.sqrt(counts[i]) / (total*spaces[i]) for i in range(len(counts))])

    txtname = outputdir + caseDir + caseDir+"_data" + '.txt'
    f = open(txtname, "w")
    for i in range(0, len(meshpoints)):
        f.write(str(meshpoints[i])+", "+str(proportion[i])+", "+str(sigma[i])+"\n")
    f.close

    txtname = outputdir + caseDir + caseDir+"_summary"+ '.txt'
    f = open(txtname, "w")
    f.write(caseName+ f" Summary, {total} Total Generations:\n")
    f.write(f"Begin time: {date_str}\n")
    valText = "k={k:.5E}\nCapture Rate= {c:.5E} s-1\nFission Rate= {f:.5E} s-1\nLeakage Rate= {l:.5E} s-1\nSource Rate= {s:.5E} s-1\n\n"
    valText= valText.format(k=keff, c=probCap, f=probFiss, l=probLeak, s=S)
    f.write(valText)

    nutxt = "{p:.5E} "
    f.write("Induced Fission Probability (0,1,2,...):\n")
    for i in range(0, len(p_nu)):
        f.write(nutxt.format(p=p_nu[i]))

    f.write("\n\nSpontaneous Fission Probability (0,1,2,...):\n")
    for i in range(0, len(nu_s)):
        f.write(nutxt.format(p=nu_s[i]))
    f.write("\n")
    f.write("\n\n\nTime        Proportion  St. Dev\n-------------------------------\n")
    datamsg = "{msh:.4e}, {prop:.4e}, {sig:.4e}\n"
    for i in range(0, len(meshpoints)):
        f.write(datamsg.format(msh=meshpoints[i], prop=proportion[i], sig=sigma[i]))
    f.close

    plt.figure(figsize=(12, 8))
    plt.errorbar(meshpoints, proportion, yerr=sigma, fmt='.', markersize=10.6, capsize=2.6666)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim((10**plotmin, 10**plotmax))
    plt.ylim((1*10**-9, 10**0))
    fig = plt.gcf()

    title = "$k = {kval:.4f}$, $p_f = {P:.4f}$, $m = {init:d}, S={source:.1e}$"
    plt.title(title.format( kval=keff, P = probFiss, init=m, source=S), fontsize=26.666)
    plt.xlabel("$t$ [$\\tau$]", fontsize=16)
    plt.ylabel("$\\rho$", fontsize=16)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.savefig(outputdir+caseDir+caseDir + '.png')

    if saveData != True:
        os.system(f"rm out/{caseName}/{caseName}-data-*.txt")