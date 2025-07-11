import sys
import ctypes
import numpy
import os
import csv
import math
import timeit
import matplotlib
import matplotlib.pyplot as plt

from mpi4py import MPI
comm = MPI.COMM_WORLD
root = 0
rank = comm.Get_rank()
num_procs = comm.Get_size()

# print(num_procs, " procs")
# print(rank, ", rank")

start_time = timeit.default_timer()


# keff = 1.0
# p_nu = [0, 0.0090, 0.0220, 0.1786, 0.3094, 0.3094, 0.1118, 0.0599]
p_nu = [0, 0, 1]
# mesh = numpy.array([0, 5, 10, 50])
m=1
S = 0.0

Gens = int(sys.argv[1])
caseName = sys.argv[2]
keff = float(sys.argv[3])
stop = float(sys.argv[4])
num = int(sys.argv[5])

logs = numpy.linspace(0, numpy.log10(stop), num)
mesh = numpy.array([0])
mesh = numpy.append(mesh, numpy.power(10, logs))


outputdir = "./out"     
caseDir = "/" + caseName

nubar = numpy.dot(p_nu, numpy.linspace(0, len(p_nu)-1, len(p_nu)))
probFiss = keff/nubar
probCap = 1-probFiss
probLeak = 0

gens_per_thread = int(numpy.ceil(Gens/num_procs))
Gens = gens_per_thread*num_procs
t_mesh_name = f"out/{caseName}/t_mesh.txt"
if rank ==root:
    if not os.path.exists(outputdir+caseDir):
        os.mkdir(outputdir+caseDir)
    
    t_mesh_file = open(t_mesh_name, "w")
    for i in mesh:
        t_mesh_file.write(str(i)+"\n")
        # print(str(i))
    t_mesh_file.close()

    nu_file = open("p_nu_f.txt", "w")
    for i in p_nu:
        nu_file.write(str(i)+" ")
    nu_file.close()

    os.system("g++ -o working.so -shared -fPIC -std=c++11 extinction.cpp")
    for i in numpy.linspace(1, num_procs-1, num_procs-1):
        comm.send(0, dest=i, tag=10*i)

if rank != root:
    f = comm.recv(source=0, tag=10*rank)

path = ctypes.CDLL("./working.so")

datapath = str.encode(f"out/{caseName}/{caseName}-data-{str(rank)}.txt")
path.numDist.argtypes = [ctypes.c_int,ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,ctypes.c_double, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
path.numDist(m, gens_per_thread, probCap, probFiss, probLeak, S, ctypes.c_char_p(datapath), ctypes.c_char_p(str.encode(t_mesh_name)), rank)

pause = timeit.default_timer()
# print("Generation time= ", pause-start_time)

reader = csv.reader(open(f"out/{caseName}/{caseName}-data-{str(rank)}.txt", "r"), delimiter=",")
x = list(reader)
data = numpy.array(x).astype("int")



ns = data[0, :]


# print(fissions)

if rank != root:
    comm.send(ns, dest=0, tag=100*rank)

if rank == root:
    listns = numpy.array(ns)

    for i in numpy.linspace(1, num_procs-1, num_procs-1):
        listns = numpy.append(listns, comm.recv(source=i, tag=100*i))
    master_ns = numpy.array([]).astype('int')
    for i in range(0, max(ns)+1):
        if numpy.sum(listns == i) != 0:
            master_ns = numpy.append(master_ns, i)
    for i in numpy.linspace(1, num_procs-1, num_procs-1):
        comm.send(master_ns, dest=i, tag=10)

if rank != root:
    master_ns = comm.recv(source=root, tag=10)

j = 0
data_height = len(data[:,0])
master_data = numpy.zeros((data_height, len(master_ns)), dtype='int')
for i in range(0, len(ns)):
    j = numpy.where(master_ns == ns[i])
    master_data[:,j] = data[:,i].reshape(data_height,1,1)

if rank != root:
    comm.send(master_data, dest=0, tag=100*rank)
if rank == root:

    # os.system("rm t_mesh.txt")
    os.system(f"rm out/{caseName}/{caseName}-data-*.txt")



    for i in numpy.linspace(1, num_procs-1, num_procs-1):
        master_data[1:,:]+=comm.recv(source=i, tag=100*i)[1:,:]

    data = master_data
    ns = master_ns

    counts = data[1:len(mesh)+1,:].astype('float32')
    fissions = data[len(mesh)+1:,:].astype('float32')

    widths = numpy.append(1, numpy.power(10, numpy.floor(numpy.log10(ns[1:]))))


    props = (1/Gens)*counts*(1/widths)
    counts[counts==0] = numpy.nan
    SD_counts = numpy.sqrt(counts)
    SD_counts = SD_counts*(1/Gens)*(1/widths)
    counts[counts==numpy.nan]=0

    fiss_props = (1/Gens)*fissions*(1/widths)
    fissions[fissions==0] = numpy.nan
    SD_fissions = numpy.sqrt(fissions)
    SD_fissions = SD_fissions*(1/Gens)*(1/widths)
    fissions[fissions==numpy.nan] = 0
    counts[counts==numpy.nan] = 0
    # SD_fissions = SD_fissions*(1/gens_per_thread)

    output_file = open(f"out/{caseName}/{caseName}_out.txt", "w")
    # for i in ns:
    #     output_file.write(str(i)+", ")
    #     # print(str(i))
    # output_file.write("\n\n")

    label_length = len(str(max(ns)))+1
    output_file.write("Time mesh:\n")
    output_file.write(" "*(label_length))
    for i in range(0, len(mesh)):
        output_file.write(f"{mesh[i]:.2e} ")
    output_file.write("\n\n")
    output_file.write("Neutron Numbers:\n")
    for j in range(0, len(ns)):
        output_file.write(f"{ns[j]}")
        output_file.write(" "*(label_length-len(str(ns[j]))))
        for i in range(0, len(mesh)):
            output_file.write(f"{props[i,j]:.2e} ")
        output_file.write("\n")
    output_file.write("\n\n")
    output_file.write("Fission Numbers:\n")
    for j in range(0, len(ns)):
        output_file.write(f"{ns[j]}")
        output_file.write(" "*(label_length-len(str(ns[j]))))
        for i in range(0, len(mesh)):
            output_file.write(f"{fiss_props[i,j]:.2e} ")
        output_file.write("\n")

    plotpoints = numpy.append(0*widths[widths==1], widths[widths!=1])
    plotpoints = ns + 0.5*plotpoints
    # print(plotpoints)

    # for i in range(0, len(mesh)):
    #     for j in range(0, len(ns)):
    #         output_file.write(str(props[i, j])+", ")
    #     output_file.write("\n")
    # output_file.write("\n\n")
    # for i in range(0, len(mesh)):
    #     for j in range(0, len(ns)):
    #         output_file.write(str(fiss_props[i, j])+", ")
    #     output_file.write("\n")
    output_file.close()

    print("Total time= ", timeit.default_timer()-start_time)


    plt.figure(figsize=(9, 6))
    for i in range(1, len(mesh)):
        plt.errorbar(ns, props[i,:], yerr=2*SD_counts[i,:], fmt='.', markersize=8, capsize=3)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Neutron number", fontsize=16)
    plt.ylabel("Probability", fontsize=16)
    plt.title(f"Pt(n), k={keff}, pf={probFiss}", fontsize=22)
    plt.savefig(outputdir+caseDir+caseDir + '_ptn.png')

    plt.figure(figsize=(9, 6))
    plt.errorbar(mesh, props[:,0], yerr=2*SD_counts[:,0], fmt='.--', markersize=10, capsize=3)
    plt.errorbar(mesh, props[:,1], yerr=2*SD_counts[:,1], fmt='.--', markersize=10, capsize=3)
    plt.errorbar(mesh, props[:,2], yerr=2*SD_counts[:,2], fmt='.--', markersize=10, capsize=3)
    
    plt.legend([ns[0], ns[1], ns[2]])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time [s]", fontsize=16)
    plt.ylabel("Probability", fontsize=16)
    plt.title(f"Pn(t), k={keff}, pf={probFiss}", fontsize=22)
    plt.savefig(outputdir+caseDir+caseDir + '_pnt.png')

    plt.figure(figsize=(9, 6))
    for i in range(1, len(mesh)):
        plt.errorbar(plotpoints, fiss_props[i,:], yerr=2*SD_fissions[i,:], fmt='.', markersize=8, capsize=2.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("# Fissions", fontsize=16)
    plt.ylabel("Probability", fontsize=16)
    plt.title(f"Fission number, k={keff}, pf={probFiss}", fontsize=22)
    plt.savefig(outputdir+caseDir+caseDir + '_fissErr.png')

    legend = [" "]*len(mesh[1:])
    for i in range(0, len(mesh)-1):
        legend[i] = f"{mesh[i+1]:.2e}"

    plt.figure(figsize=(9, 6))
    for i in range(1, len(mesh)):
        plt.scatter(plotpoints, fiss_props[i,:])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("# Fiss", fontsize=16)
    plt.ylabel("Probability", fontsize=16)
    plt.legend(legend)
    plt.title(f"Fission number, k={keff}, pf={probFiss}", fontsize=22)
    plt.savefig(outputdir+caseDir+caseDir + '_fiss.png')