import os
import logging

import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation

from scipy.optimize import minimize
from scipy.sparse import csr
from astropy.cosmology import Planck15
from astropy.units import Mpc, km, s

from nbodykit import CurrentMPIComm
from nbodykit.dataset import DataSet
from nbodykit.meshtools import SlabIterator
from pmesh.pm import ComplexField
from nbodykit.source.mesh import LinearMesh
from nbodykit.source.mesh import BigFileMesh
from nbodykit.io.bigfile import BigFile


class FGPA(object):
    logger = logging.getLogger('FGPA')
    
    def __init__(self, source, z, seed, Nmesh, BoxSize, smoothing):
        # setting parameters
        self.attrs = {}
        self.cosmo = Planck15
        self.attrs['z'] = z
        self.attrs['seed'] = seed
        self.attrs['smoothing'] = smoothing
        self.attrs['BoxSize'] = BoxSize

        # import data from simulation
        mesh = source.to_mesh(Nmesh=Nmesh, BoxSize=BoxSize)
        real = mesh.paint(mode='real')
        self.pm = real.pm

        self.attrs['BoxSize'] = mesh.pm.BoxSize
        self.attrs['Nmesh'] = mesh.pm.Nmesh

        # apply Gaussian smoothing to real field to get baryons instead of DM
        # smoothing = 0.07: random parameter that makes sense
        # (100 kpc in bi and davidsen http://iopscience.iop.org/article/10.1086/303908/pdf)
        real = real.r2c().apply(lambda k, v: v * np.exp(- 0.5 * self.attrs['smoothing'] ** 2 * sum(ki **2 for ki in k))).c2r()

        # calculate overdensity field  
        delta_mean = real.cmean()
        self.delta = real / delta_mean - 1

        # calculate optical depth and flux (with and without RSD and thermal broadening) along a skewer in z direction

        vel_to_dist = (self.attrs['z'] + 1) / (self.cosmo.efunc(self.attrs['z']) * 100.0)
        self.vel_to_dist = vel_to_dist
    
        # calculate velocity field
        source['vz'] = source['Velocity'][:, 2]
        # XXX : velocity field is not smoothed. Is this desirable?
        vreal_accum = source.to_mesh(Nmesh=self.attrs['Nmesh'], BoxSize=self.attrs['BoxSize'] , weight='vz').paint(mode='real')
        self.velocity = np.where(real[:] != 0., vreal_accum[:] / real[:], 0)

        # calculate RSD
        self.dreal = np.linspace(0, self.attrs['BoxSize'][2], self.attrs['Nmesh'][2], endpoint=False)


    def fit(self, N, seed, T_IGM, A0=0.6, gamma0=2.6):
        # call generate_los
        # compute
        # iterate till  agrees with the model
        skewer_pos = self.create_los(N, seed)

        def model(A, gamma):
            cat = FGPASkewerCatalog(self, skewer_pos, A=A, T_IGM=T_IGM, gamma=gamma)
            F_red = cat.fields['F_red']
            F_sum = self.pm.comm.allreduce(np.sum(F_red))
            F2_sum = self.pm.comm.allreduce(np.sum(F_red * F_red))
            N = self.pm.comm.allreduce(F_red.size)
            F_mean = F_sum / N
            F_var = F2_sum / N - F_mean ** 2
            self.logger.info("Iteration at %g %g : F_mean= %g F_var=%g" % (A, gamma, F_mean, F_var))
            return F_mean, F_var

        def objective(x):
            F_mean, F_var = model(x[0], x[1])
            F_mean_fiducial, F_var_fiducial = flux_fiducial(self.attrs['z'])
            chi_squared_F_mean = (F_mean_fiducial - F_mean)**2/F_mean_fiducial**2
            chi_squared_F_var = (F_var_fiducial - F_var)**2/F_var_fiducial**2
            return chi_squared_F_mean  + chi_squared_F_var

        res = minimize(objective, (A0, gamma0), method = 'Nelder-Mead')
        A, gamma = res.x[0], res.x[1]
        
        return A, gamma

    def create_los(self, N, seed):
        rng = np.random.RandomState(seed)

        pm = self.pm
        
        if pm.comm.rank == 0:
            pos = (rng.uniform(size = (N, 3)) * self.attrs['BoxSize']).astype('f8')
            pos[:, -1] = 0
        else:
            pos = np.empty((0, 3), dtype='f8')
	
#	print(pos[:][:5])
        return pos

SQRT_KELVIN_TO_KMS = 0.12849 #sqrt(2 * k_B / m_proton) in km/s

class FGPASkewerCatalog(object):
    def __init__(self, fgpa, skewer_pos, A, T_IGM, gamma):
        self.attrs = {}
        self.cosmo = fgpa.cosmo
        self.attrs['T_IGM'] = T_IGM
        self.attrs['gamma'] = gamma
        self.attrs['A'] = A
        self.attrs['z'] = fgpa.attrs['z']

        pm = fgpa.pm
        layout = pm.decompose(skewer_pos, smoothing=0)
        self.skewer_pos = layout.exchange(skewer_pos)

        data = np.empty((len(self.skewer_pos), len(fgpa.dreal)), dtype=[
                ('delta', 'f8'),
                ('tau_real', 'f8'),
                ('tau_red', 'f8'),
                ('F_real', 'f8'),
                ('F_red', 'f8'),
                ('delta_F', 'f8')])

        for i, pos in enumerate(self.skewer_pos):
            ix, iy, iz = np.int32(pm.affine.scale * pos + pm.affine.translate)
            #print(ix, iy, iz)
            d = self.one_line_of_sight((ix, iy), fgpa.delta, fgpa.velocity, fgpa.dreal, fgpa.vel_to_dist)
            for key, value in d.items():
                data[key][i] = value

        self.pm = pm
        self.dreal = fgpa.dreal
        self.fields = data

    def one_line_of_sight(self, xy_skewer_pos, delta, velocity, dreal, vel_to_dist):
        
        # pick a skewer out of the simulated 3D box
        xy_skewer_pos = tuple(xy_skewer_pos)
        delta_z = delta[xy_skewer_pos]
        velocity_z = velocity[xy_skewer_pos]
        T_z = self.attrs['T_IGM'] * ( 1 + delta_z ) ** (self.attrs['gamma'] - 1.)

        dRSD = velocity_z * vel_to_dist
        dred = dreal + dRSD

        # calculate thermal broadening
        vtherm = SQRT_KELVIN_TO_KMS * np.sqrt(T_z)
        dtherm = vtherm * vel_to_dist
        
        # calculate optical depth tau_real from density with fluctuating Gunn-Peterson approximation
        tau_real_z = fgpa(delta_z, self.attrs['A'], self.attrs['gamma'])
        
        # calculate optical depth with RSD and thermal broadening using funky convolution with Gaussians
        tau_red_z = irconvolve(dreal, dred, tau_real_z, dtherm)
        
        # calculate flux from optical depth
        F_real_z = np.exp(- tau_real_z)
        F_red_z = np.exp(- tau_red_z)
        F_mean_fiducial = flux_fiducial(self.attrs['z'])[0]
        F_red_normalized_z = F_red_z / F_mean_fiducial - 1

        return {'delta':delta_z,
                'tau_real':tau_real_z,
                'tau_red':tau_red_z,
                'F_real':F_real_z,
                'F_red':F_red_z,
                'delta_F':F_red_normalized_z}


    @classmethod
    def load(kls, filename, root):
        pass

    def save(self, filename, root):
        from bigfile import BigFileMPI
        with BigFileMPI(self.pm.comm, filename, create=True) as ff:
            path = root + '/' + 'skewer_pos'
            with ff.create_from_array(path, self.skewer_pos) as bb:
                pass
            for key in sorted(self.fields.dtype.fields):
                path = root + '/' + key
                data = self.fields[key]
                with ff.create_from_array(path, data) as bb:
                    pass
 
def sample_PS(fgpa_obj, which_field, number_of_skewers, random_seed):
    F_mean_fiducial = flux_fiducial(fgpa_obj.attrs['z'])[0]
    los = create_los(fgpa_obj, which_field, number_of_skewers, randon_seed, F_mean_fiduacial)
    PS_1d = power_spectrum_1d(los, fgpa_obj.attrs['z'], fgpa_obj.attrs['BoxSize'], fgpa_obj.attrs['Nmesh'])
    return PS_1d

# calculate fiducial values for flux and variance of flux
def flux_fiducial(z):
    F_mean_fiducial = np.exp(-0.001845*(1 + z)**3.924)
    F_var_fiducial = 0.065*((1 + z)/3.25)**3.8 * F_mean_fiducial**2
    return F_mean_fiducial, F_var_fiducial

# calculate 1d power spectrum
def power_spectrum_1d(fgpa_obj, los):
    pm = fgpa_obj.pm
    Nlos = pm.comm.allreduce(len(los))
    print('---------------', Nlos)
    los_fft = np.fft.rfft(los, axis = 1) / los.shape[1]
    los_fft_abs_sqr = np.absolute(los_fft) ** 2 
    
    vel_to_dist = (fgpa_obj.attrs['z'] + 1) / Planck15.H(fgpa_obj.attrs['z']) / (Mpc/Planck15.h) * km / s

    PS_1d = pm.comm.allreduce(np.sum(los_fft_abs_sqr, axis = 0)) * (fgpa_obj.attrs['BoxSize'][2]/vel_to_dist) 
    PS_1d /= Nlos
    return PS_1d


def create_los(fgpa_obj, which_field, number_of_skewers, random_seed):
    np.random.seed(random_seed)
    xy_skewer_pos_array = np.int32(np.random.uniform(size = (number_of_skewers,2)) * fgpa_obj.attrs['Nmesh'][2])
    
    los = np.zeros((len(xy_skewer_pos_array), fgpa_obj.attrs['Nmesh'][2]))
    for i in range(len(xy_skewer_pos_array)):
        los[i] = line_of_sight(fgpa_obj, xy_skewer_pos_array[i])[which_field]
    return los


# Fluctuating Gunn-Peterson approximation to obtain optical depth from density field
def fgpa(delta, A, gamma):
    tau = A * (1 + delta) ** ( 2 - 0.7 * (gamma - 1) )
    return tau
    
    
# Yu's fancy convolution function
def irconvolve(xc, x, y, h, 
        kernel=lambda r, h: np.exp(- 0.5 * (r / h) ** 2)):
    """ default kernel is gaussian
        exp - 1/2 * r / h
        xc has to be uniform!
    """
    xc, y, x, h = np.atleast_1d(xc, y, x, h)
    dxc = (xc[-1] - xc[0]) / (len(xc) - 1)
    support = 6

    #first remove those are too far off
    good = ((x + support * h > xc[0]) \
          & (x - support * h < xc[-1]))
    x = x[good]
    y = y[good]
    h = h[good]

    if len(h) > 0:
        # the real buffer is bigger than out to ease the normalization
        # still on the edge we are imperfect
        padding = int((2 * support + 1)* h.max() / dxc) + 1
        padding = max(padding, 2)
        buffer = np.zeros(shape=len(xc) + 2 * padding)
        paddedxc = np.empty(buffer.shape, dtype=xc.dtype)
        paddedxc[padding:-padding] = xc
        # here comes the requirement xc has to be uniform.
        paddedxc[:padding] = xc[0] - np.arange(padding, 0, -1) * dxc
        paddedxc[-padding:] = xc[-1] + np.arange(1, padding +1) * dxc
        out = buffer[padding:-padding]
        assert len(out) == len(xc)
        assert (paddedxc[1:] > paddedxc[:-1]).all()

        # slow. for uniform xc/paddedxc, we can do this faster than search
        start = paddedxc.searchsorted(x - support * h, side='left')
        end = paddedxc.searchsorted(x + support * h, side='left')
        # tricky part, build the csr matrix for the conv operator,
        # only for the non-zero elements (block diagonal)
        N = end - start + 1
        indptr = np.concatenate(([0], N.cumsum()))
        indices = np.repeat(start - indptr[:-1], N) + np.arange(N.sum())
        r = np.repeat(x, N) - paddedxc[indices]
        data = kernel(r, np.repeat(h, N))
        data[np.repeat(N==1, N)] = 1
        data[np.repeat(h==0, N)] = 1
        matrix = csr.csr_matrix((data, indices, indptr), 
                shape=(len(x), len(paddedxc)))
        norm = np.repeat(matrix.sum(axis=1).flat, N)
        data /= norm
        buffer[:] = matrix.transpose() * y
    else:
        out = np.zeros(shape=xc.shape, dtype=y.dtype)
    return out


def fitting_method(fgpa_obj, A_initial, gamma_initial, which_field, number_of_skewers, random_seed, T_IGM):

    def model(A, gamma, fgpa_obj, which_field, number_of_skewers, random_seed, T_IGM):
        fgpa_obj.set_A(A)
        fgpa_obj.temperature_field(fgpa_obj, T_IGM, gamma)
        F = create_los(fgpa_obj, which_field, number_of_skewers, random_seed)
        F_mean = np.mean(F)
        F_var = np.var(F)
        return F_mean, F_var

    def objective(x, which_field, number_of_skewers, random_seed, fgpa_obj, T_IGM):
        F_mean, F_var = model(x[0], x[1], fgpa_obj, which_field, number_of_skewers, random_seed, T_IGM)
        F_mean_fiducial, F_var_fiducial = flux_fiducial(fgpa_obj.attrs['z'])
        chi_squared_F_mean = (F_mean_fiducial - F_mean)**2/F_mean_fiducial**2
        chi_squared_F_var = (F_var_fiducial - F_var)**2/F_var_fiducial**2
        return chi_squared_F_mean  + chi_squared_F_var

    res = minimize(objective, (A_initial,gamma_initial),
                   args = (which_field, number_of_skewers, random_seed, T_IGM),
                   method = 'Nelder-Mead')
    A, gamma = res.x[0], res.x[1]
    
    return A, gamma
    

def fit(simulation_file, boxsize, number_of_pixels, z, smoothing, T_IGM,
                    A_initial, gamma_initial, which_field, number_of_skewers, random_seed):
    
    F_mean_fiducial, F_var_fiducial = flux_fiducial(z)

    particle_pos, particle_vel = import_data(simulation_file)
    delta, velocity, T_junk = painting(particle_pos = particle_pos, particle_vel = particle_vel,
                                  boxsize = boxsize, number_of_pixels = number_of_pixels,
                                  smoothing = smoothing, T_IGM = T_IGM, gamma = gamma_initial)
    A, gamma = fit(A_initial, gamma_initial, which_field, number_of_skewers, random_seed,
                   delta, velocity, z, boxsize, number_of_pixels, F_mean_fiducial, F_var_fiducial, T_IGM)
    return A, gamma

