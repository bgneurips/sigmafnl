import json
import fire
import numpy
import logging
from rich import print
from rich.logging import RichHandler
from matplotlib import pyplot as plt
from classy import Class
from scipy import interpolate
from matplotlib import pyplot as plt

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger(__name__)

def plot_ellipse(F, bg, fnl, ls='-', label=None, color='k'):
    
    sigmax = numpy.round(numpy.sqrt(numpy.linalg.inv(F))[1,1], 4).real
    sigmay = numpy.round(numpy.sqrt(numpy.linalg.inv(F))[0,0], 4).real
    sigmaxy = numpy.round((numpy.linalg.inv(F))[1,0], 4).real

    width = numpy.sqrt((sigmax**2 + sigmay**2)/2 + numpy.sqrt((sigmax**2 - sigmay**2)**2/4 + sigmaxy**2))
    height = numpy.sqrt((sigmax**2 + sigmay**2)/2 - numpy.sqrt((sigmax**2 - sigmay**2)**2/4 + sigmaxy**2))
    angle = numpy.degrees(0.5*numpy.arctan(2*sigmaxy/(sigmax**2 - sigmay**2)))

    from matplotlib.patches import Ellipse

    ell = Ellipse(xy=(fnl, bg), width=2.48*width, height=2.48*height, angle=angle, ls=ls, label=label, color=color, fill=False)
    return ell


def alpha_and_powerspectrum(k):

    with open('/home/ugiri/sigma-fnl/scripts/oldies/class_quijote.json', 'r') as f:
        class_parameters = json.load(f)
        class_parameters['z_pk'] = 0
        del class_parameters['A_s']
        class_parameters['sigma8'] = 0.834
    model = Class()
    model.set(class_parameters)
    model.compute()
    transfer = model.get_transfer(0)

    factor = 1/model.h()
    alpha = interpolate.interp1d(transfer['k (h/Mpc)']*model.h()*factor,
            transfer['d_tot'], fill_value='extrapolate', bounds_error=False)
    powerspectrum = numpy.zeros(10000)
    for i,kval in enumerate(numpy.geomspace(1e-4, 1e2, 10000)):
        powerspectrum[i] = model.pk(kval/factor, 0)/factor**3
    powerspectrum = interpolate.interp1d(numpy.geomspace(1e-4, 1e2, 10000), powerspectrum, fill_value='extrapolate')

    return alpha(k), powerspectrum(k)


def analyticDensityFisher(bg, beta, fnl, shot_noise, k):
    beta *= 2
    Ffnlfnl, Ffnlbg, Fbgbg = 0, 0, 0
    covariance = numpy.zeros((2,2), dtype=numpy.float64);
    alpha, pmm = alpha_and_powerspectrum(k)
    for i in range(0, len(k)):
        covariance[0,0] = (bg + fnl*beta/alpha[i])**2 * pmm[i] + shot_noise
        covariance[1,1] = pmm[i]
        covariance[0,1] = (bg + fnl*beta/alpha[i]) * pmm[i]
        covariance[1,0] = (bg + fnl*beta/alpha[i]) * pmm[i]

        cinv = numpy.linalg.inv(covariance)

        dCdbg = numpy.array([[(2*bg)*pmm[i], pmm[i]], [pmm[i], 0]])
        dCdbg = numpy.array([[(2*bg)*pmm[i], pmm[i]], [pmm[i], 0]])
        dCdfnl = numpy.array([[(2*bg*beta/alpha[i])*pmm[i], pmm[i]*beta/alpha[i]], [pmm[i]*beta/alpha[i], 0]])

        Ffnlfnl += 0.5*numpy.trace(numpy.matmul(numpy.matmul(numpy.matmul(cinv, dCdfnl), cinv), dCdfnl))
        Fbgbg += 0.5*numpy.trace(numpy.matmul(numpy.matmul(numpy.matmul(cinv, dCdbg), cinv), dCdbg))
        Ffnlbg += 0.5*numpy.trace(numpy.matmul(numpy.matmul(numpy.matmul(cinv, dCdfnl), cinv), dCdbg))

    return numpy.array([[Fbgbg, Ffnlbg], [Ffnlbg, Ffnlfnl]])


def main(bg=1.406, beta=0.58, fnl=0, shot_noise=2577, filename='fisher_matter+halo.pdf'):

    modes = numpy.load('../analysis/modes_fnl=0_fiducial_n=20.npz')
    k = numpy.repeat(modes['k'].reshape(100,30)[0,:30].squeeze(), 100)
    
    Fmm = analyticDensityFisher(bg, beta, fnl, shot_noise, k)
    logger.info('Error on fnl using halo and density field: {}'.format((numpy.sqrt (numpy.linalg.inv(Fmm))[1,1])))
    logger.info('Error on bg using halo and density field: {}'.format((numpy.sqrt(numpy.linalg.inv(Fmm))[0,0])))   

    _, ax = plt.subplots(dpi=200, figsize=(4,4))
    ax.add_patch(plot_ellipse(Fmm, bg, fnl,  ls='--', label=r'$[\delta_h, \delta_m]$', color='b'))
    ax.autoscale()
    ax.legend(frameon=False, loc=2)
    plt.savefig(filename, dpi=400)

if '__main__' == __name__:
    fire.Fire(main)