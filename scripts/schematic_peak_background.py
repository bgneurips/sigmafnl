import numpy
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
numpy.random.seed(41)

deltal = numpy.sin(numpy.linspace(0, 2*numpy.pi, 100))
deltas = numpy.random.normal(scale=1, size=100)


fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(6,3), dpi=400, sharey=True, sharex=True)#, gridspec_kw = {'wspace':0.05, 'hspace':0.05})
ax[0,0].axhline(1.42, ls='--', alpha=0.4)
ax[0,0].text(-4, 1.2, r'$\delta_{cr}$', size=8, math_fontfamily='dejavuserif')
ax[0,0].plot(numpy.linspace(0, 6*numpy.pi, 100), deltal+ gaussian_filter(deltas, 1.), color='k')
ax[0,1].plot(numpy.linspace(0, 6*numpy.pi, 100), deltal, color='#bb3e03')
ax[0,2].plot(numpy.linspace(0, 6*numpy.pi, 100), gaussian_filter(deltas, 1.), color='#0077b6' )
ax[0,0].axis('off')
ax[0,1].axis('off')
ax[0,2].axis('off')
ax[0,0].text(9,-4, r'$\phi$', size=10, math_fontfamily='dejavuserif')
ax[0,0].text(20.7, 0, r'$=$', size=10, math_fontfamily='dejavuserif')
ax[0,1].text(8,-4, r'$\phi_l$', size=10, math_fontfamily='dejavuserif')
ax[0,1].text(20.7, 0, r'$+$', size=10, math_fontfamily='dejavuserif')
ax[0,2].text(5.5,-4, r'$\phi_s$', size=10, math_fontfamily='dejavuserif')
ax[0,1].set_title(r'$f_{NL} = 0$', math_fontfamily='dejavuserif')

ax[1,0].axhline(1.42, ls='--', alpha=0.4)
ax[1,0].text(-4, 1.2, r'$\delta_{cr}$', size=8, math_fontfamily='dejavuserif')
ax[1,0].plot(numpy.linspace(0, 6*numpy.pi, 100), (deltal + (1+2*deltal)*gaussian_filter(deltas, 1.)), color='k')
ax[1,1].plot(numpy.linspace(0, 6*numpy.pi, 100), deltal, color='#bb3e03')
ax[1,2].plot(numpy.linspace(0, 6*numpy.pi, 100), ((1+2*deltal)*gaussian_filter(deltas, 1.)), color='#0077b6')
ax[1,0].axis('off')
ax[1,1].axis('off')
ax[1,2].axis('off')
ax[1,0].text(9,-4, r'$\phi$', size=10, math_fontfamily='dejavuserif')
ax[1,0].text(20.7, 0, r'$=$', size=10, math_fontfamily='dejavuserif')
ax[1,1].text(8,-4, r'$\phi_l$', size=10, math_fontfamily='dejavuserif')
ax[1,1].text(20.7, 0, r'$+$', size=10, math_fontfamily='dejavuserif')
ax[1,2].text(5.5,-4, r'$\phi_s \rightarrow (1+f_{NL}\phi_l)\phi_s$', size=10, math_fontfamily='dejavuserif')
ax[1,1].set_title(r'$f_{NL} > 0$', math_fontfamily='dejavuserif')
#plt.margins(0.005, tight=True)
#fig.suptitle('$peak-background \ split$')
fig.tight_layout()
plt.savefig('pbs.pdf')