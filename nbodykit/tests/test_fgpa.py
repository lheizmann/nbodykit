from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_array_equal, assert_allclose
from nbodykit.algorithms.fgpa_03 import *
from nbodykit.source.mesh.linear import LinearMesh
import numpy as np

# debug logging
setup_logging("debug")
    
@MPITest([1, 4])
def test_fgpa(comm):
    import matplotlib.pyplot as plt
    CurrentMPIComm.set(comm)
    cosmo = cosmology.Planck15

#    source = BigFileCatalog('/home/lukas/source/nbodykit/nbodykit/algorithms/nbodykit_testrun_5Mpc_128p_z3_seed100_0.2500', dataset='1/', header='Header')
    source = LogNormalCatalog(cosmology.NoWiggleEHPower(cosmo, 3.), nbar=1024, BoxSize=5.0, bias=1.0, Nmesh=128., seed=33)

    #fgpa = FGPA(source, 3, 1.0, 1.0, 100, 0, Nmesh = 128, BoxSize = 5., smoothing = 0.07, T_IGM = 1e4)
    #fgpa = FGPA(source, 3, 100, 128, 5., 0.07, 0.60454020383942408, 2.6732609287058589, 1e4, 100)
    fgpa = FGPA(source, z = 3, seed = 100, Nmesh = 128, smoothing=0.07)

    A, gamma = fgpa.fit(N=100, seed=100, T_IGM=1e4)

    skewers = fgpa.create_los(N=1000, seed=100)
    cat = FGPASkewerCatalog(fgpa, skewers, T_IGM=1e4, gamma=2.67326, A=0.60454)
    cat = FGPASkewerCatalog(fgpa, skewers, T_IGM=1e4, gamma=gamma, A=A)

    POW = power_spectrum_1d(fgpa, cat.fields['delta_F'])
   # assert_array_equal(fgpa.tau_red, 0)
    #assert_array_closeto(fgpa.overdensity, 3.4, tol=1e-02)

    vel_to_dist = (3 + 1) / Planck15.H(3) / (Mpc/Planck15.h) * km / s
    k = 2*np.pi / (5./vel_to_dist) * np.arange(128/2 + 1)
    
    cat.save("test_fgpa", "sample-%d" % comm.size)
    if comm.rank == 0:
        plt.plot(k, k * POW / np.pi, label = 'F_red_normalized, 5Mpc')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('k in (km/s)^-1')
        plt.ylabel('k*P(k)/pi')
        plt.xlim([5e-3,1])
        #plt.ylim([7e-5,1e-1])
        plt.legend()
        plt.savefig('foo-%d.png' % comm.size)

