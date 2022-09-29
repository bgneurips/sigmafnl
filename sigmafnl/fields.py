import numpy
from nbodykit.lab import *

class Field:

    def __init__(self, field, boxsize=None):

        self.field = field
        self.boxsize = boxsize
        
        self.pixels = self.field.shape[0]
        self.dimensions = sum([x for x in self.field.shape if x !=1])
        
    @property
    def _mesh(self):
        return ArrayMesh(self.field, BoxSize=self.boxsize)

    def powerspectrum(self, dk=None):
        powerspectrum = FFTPower(self._mesh, mode='1d', dk=dk)
        return {'k': powerspectrum.power['k'][:].real, 'powerspectrum': powerspectrum.power['power'][:].real}

    
    def cross_spectrum(self, second, dk=None):
        
        if isinstance(second, numpy.ndarray):
            second = Field(second, boxsize=self.boxsize)
        powerspectrum = FFTPower(first=self._mesh, second=second._mesh, mode='1d', BoxSize=self.boxsize, dk=dk)
        return {'k': powerspectrum.power['k'][:].real, 'powerspectrum': powerspectrum.power['power'][:].real}


    def cross_correlation(self, second, dk=None):
        
        if isinstance(second, numpy.ndarray):
            second = Field(second, boxsize=self.boxsize)

        cpk = self.cross_spectrum(second=second, dk=dk)
        opk = self.powerspectrum(dk=dk)['powerspectrum']
        spk = second.powerspectrum(dk=dk)['powerspectrum']

        r = cpk['powerspectrum']/(numpy.sqrt(opk)*numpy.sqrt(spk))
        
        return {'k': cpk['k'], 'r': r}

    def largest_n_modes(self, n):

        k = 2*numpy.pi*numpy.fft.fftfreq(n=self.pixels, d=self.boxsize/self.pixels)
        kx, ky, kz = numpy.meshgrid(k, k, k[:len(k)//2+1])
        kgrid = numpy.sqrt(kx**2 + ky**2 + kz**2)
        k = kgrid.flatten()
        args = numpy.argsort(k)[:n]

        return {'k': k[args], 'field_modes': numpy.fft.rfftn(self.field).flatten()[args]}

    def kmod(self):
        k = 2*numpy.pi*numpy.fft.fftfreq(n=self.pixels, d=self.boxsize/self.pixels)
        kx, ky, kz = numpy.meshgrid(k, k, k[:len(k)//2+1])
        kgrid = numpy.sqrt(kx**2 + ky**2 + kz**2)
        return kgrid
        
    def projection(self, percent=100):
        return self.field[:,:,:int(self.pixels*(percent/100))].sum(2)


    def relative_bias(self, second, dk=None):
        
        if isinstance(second, numpy.ndarray):
            second = Field(second, boxsize=self.boxsize)

        cpk = self.cross_spectrum(second=second, dk=dk)
        pk = self.powerspectrum()

        return numpy.mean(cpk['powerspectrum'][1:6]/pk['powerspectrum'][1:6])

    def bias_scaling(self, second, dk=None):
        
        if isinstance(second, numpy.ndarray):
            second = Field(second, boxsize=self.boxsize)

        cpk = self.cross_spectrum(second=second, dk=dk)
        pk = self.powerspectrum(dk=dk)

        return {'k': pk['k'][1:], 'bias': cpk['powerspectrum'][1:]/pk['powerspectrum'][1:]}


