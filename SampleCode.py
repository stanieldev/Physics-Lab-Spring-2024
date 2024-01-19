#!/usr/bin/env python
# coding: utf-8


# # This provides much template Python code that you will find useful.  You can copy the necessary code bits out to your local Python script, or Jupyter Notebook (local, VMware, or CoLab), or your favorite environment.  You must actively record what you do during lab periods in your OneNote notebook!


# if you want more functionality in your code, this is where you need to import packages/modules
# lmfit is the most critical package you need this semester!


from IPython.display import Latex
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from pylab import *
from math import *
from scipy.optimize import curve_fit
from lmfit import Model, Parameters
from scipy.constants import *



#Fit data to a simple gaussian
#set seeds for fit parameters (nominal values)
Amp=175 # peak counts
mu=660 # peak mean location
RMS=10 # peak RootMeanSquare


# * A large number of pre-defined fit models are included in lmfit.  You can also define your own custom fit functions.  See reference: https://lmfit.github.io/lmfit-py/builtin_models.html


def gauss(v,a,mean,sigma):
    return a*np.exp(-(v-mean)**2/(2*sigma**2))


# This is Lmfit
gModel = Model(gauss)
gModel.set_param_hint('a', value=Amp, min=0, max=Inf)
gModel.set_param_hint('mean', value=mu, min=0, max=Inf)
gModel.set_param_hint('sigma', value=RMS, min=0, max=Inf)
params = gModel.make_params()
result = gModel.fit(Intensity, v=Lambd, a=Amp, mean=mu, sigma=RMS, weights = 1.0/(D_intensity))
dely = result.eval_uncertainty(sigma=1)
plt.fill_between(Lambd, result.best_fit-dely, result.best_fit+dely, color="#ABABAB")


print('')
print('Lmfit Model fitting results:')
print(result.fit_report())               
plt.errorbar(Lambd,Intensity,yerr=D_intensity,fmt='.')
plt.plot(Lambd, result.best_fit, 'r-')
plt.title('Gaussian Fit to emission spectrum')
plt.ylabel('intensity')
plt.xlabel('wavelength (nm)')
plt.savefig('Gauss_640.png')


# ### Lmfit chi-square values are correct ONLY when you supply the " weights = 1.0/(error_bars) " parameter to the fit function!


# Here are all the gettable values after lmfit has successfully fitted your data: https://lmfit.github.io/lmfit-py/fitting.html


# We will also analyze p-value this semester:
#  Get what we need for p-value from lmfit: 
chi2 = result.chisqr 
print('Total chi2: ',chi2) 
print('Number of data points: ',result.ndata) 
print('Number of fit parameters: ',result.nvarys) 
print('Degrees of freedom: ',result.nfree) 
redChi2 = result.redchi 
print('Reduced chi2: ',redChi2) 
print('Corresponding p-value: ',1.000-sp.stats.chi2.cdf(chi2,result.nfree)) 


# 
# ### Now try out a Skewed Voigt Fit.  The Voigt distribution is used primarily in spectroscopy to explain data that has more than one smearing source.  See https://reference.wolfram.com/language/ref/VoigtDistribution.html
# 
# ### This model will return a "center" fit parameter.  When the distribution is symmetric, that value corresponds to the peak wavelength.  However, when the spectral line is asymmetric, the "center" is more like "what the wavelength would have been if this distribution were not skewed".  For our purposes, we will stick to later analysis that uses the peak wavelength.


from lmfit.models import SkewedVoigtModel
model = SkewedVoigtModel()


====================================================


# set initial parameter guesses from earlier gaussian fit. add a guess for gamma (skewness)
params = model.make_params(amplitude=best_vals[0], center=best_vals[1], sigma=best_vals[2], gamma=best_vals[2])


# adjust parameters to best fit data
result = model.fit(Intensity, params, x=Lambd, weights = 1.0/(δ_intensity))


chi2 = result.chisqr 
print('Total chi2: ',chi2) 
print('Number of data points: ',result.ndata) 
print('Number of fit parameters: ',result.nvarys) 
print('Degrees of freedom: ',result.nfree) 
redChi2 = result.redchi 
print('Reduced chi2: ',redChi2) 
print('Corresponding p-value: ',1.000-sp.stats.chi2.cdf(chi2,result.nfree)) 


plt.errorbar(Lambd, Intensity,yerr=δ_intensity,fmt='.')
plot(Lambd, result.best_fit) 
plt.title('Skewed Voigt Fit to RED emission spectrum')
plt.ylabel('intensity')
plt.xlabel('wavelength (nm)')
plt.savefig('skewVoigt_640.png')


# also report peak wavelength (max from fit curve)
print('max of curve:',np.max(result.best_fit))
peakLambda_red640 = Lambd[np.where(result.best_fit==np.max(result.best_fit))]
print(' at ',peakLambda_red640,'nm')


# store fit results for later use
λ640 = result.params['center'].value
σ640 = result.params['sigma'].value








# ## Reflections on the first vial data set
# * Before Friday's lecture, simply put, reduced $\chi^2$ should be about 1 for 'good agreement' between your fit model and your data.  This is a good rule of thumb for interpreting chi-square values for a single 'goodness of fit' question.  
# * If we are comparing two or more fit models, the model with the lowest chi-square value (given the same number of fit parameters) is generally superior.  You should gain at least a reduction of 1.0 in the reduced $\chi^2$ for every additional fit parameter that you introduce... although if you had to add too many fit parameters to gain that improvement, it's not useful for a physically-motivated description or model for you data.


# ## Next Up: 
# 1. We have collected data for six vials and fit the data with a reasonable model for one data set.  We need to repeat the two model fits for the other five data sets, decide which model to use for each, extract the peak wavelength as lambda, and error bars on lambda.  Really, we need to account for the range of dot sizes, which is described by FWHM.
# 2. Explore the relationship between wavelength and diameter, as defined in the five models of the PreLab!  We need to plot wavelength versus dot diameter D, with accurate error bars on $\lambda$.  




# ## Super Key: Python dictionaries...this is how you access Lmfit parameters!
#print(result.params['center'].value)


# * Note: 'center' wavelengths are defined in the Skewed Voigt function, and do not correspond to 'peak' or 'most probable' wavelength unless the distribution is symmetric!


# * We will use sigma as our starting best estimate for error bars on center wavelengths when we construct our $\lambda$ vs. $D$ plot and fits.


# ### Make the wavelength vs. diameter plot


# ### We will use peak wavelength for our comparison to the PreLab models, but you can see they do not differ much from the 'center' wavelength given by the Skewed Voigt fit. 


# ### Keep in mind that we are not fitting our models to this diameter data!  All parameters in our PreLab models are fixed by the physical theory, so we must simply plot each model and compute $\chi^2$ agreement with our data.  Then we can select the best model by comparing the $\chi^2$ values.


# simplest chi-square calculator from Modern Lab sample code.  ONLY USE THIS when you are comparing two sets of data, or data to some fixed model.  DO NOT USE THIS if you are performing a fit of a model function to your data - rely on lmfit to report it!
chi_squared = np.sum((func(xdata, *parameters) - ydata)**2/yerror**2)
red_chi_squared = chi_squared/(np.size(ydata)-np.size(parameters))


D_data = [2.3,2.6,3.3,4.6,5.6,6.9] # dot diamters (nm)
d_D_data = [0.1,0.1,0.1,0.1,0.1,0.1] # estimate NN-labs uncertainty, but isn't describing spread of size distribution


# define reduced mass of electron-hole system
mu = (me*mh)/(me+mh)


# define each model from our PreLab theory work.  Each mN is a function lambda(D)
def m1(d):
    return (8 * m_e * c * (d*1e-9)**2 / h) *1e9 # convert to nm


def m2(d):
    return (8/3 * m_e * c * (d*1e-9)**2 / h) *1e9 # convert to nm


def m3(d):
    return (h * c / (E_g +(3 * h**2 / (8 * m_e * (d*1e-9)**2) ))) *1e9 # convert to nm


def m4(d):
    return (h * c / (E_g +(  h**2 / (2 * m_e * (d*1e-9)**2) ))) *1e9 # convert to nm


def m5(d):
    return (h * c / (E_g + (h**2 / (2 * mu  * (d*1e-9)**2) ))) *1e9 # convert to nm


# model from "ESI The Owner Societies 2013" adds exciton binding energy.  Shows wacky ~few micron for the largest 3, 
#  and is crazy for the smallest three dots, where the binding energy becomes larger than the other two terms, 
#  causing negative wavelengths!
#def m6(d):
#    E_binding = 1.8*e**2/(6.2*epsilon_0)/(d/2*1e-9)
#    return (h * c / (E_g - E_binding + ((h**2 / (2*(d*1e-9)**2)) * (me+mh)/(me*mh)))) *1e9 # convert to nm




plt.errorbar(D, lambdas,yerr=δ_lambda,fmt='.',label='data')
plot(D, m1(D),label='m1')
plot(D, m2(D),label='m2')
plot(D, m3(D),label='m3')
plt.title('Emitted photon wavelength vs. quantum dot diameter')
plt.ylabel('$\lambda$ (nm)')
plt.xlabel('D (nm)')
plt.legend()
plt.savefig('CdSe_dotModel.png')




# ### Clearly, the first two models (infinite wells in vacuum) are garbage.  Let's compare models 3, 4, and 5:


plt.errorbar(D, lambdas,yerr=δ_lambda,fmt='.',label='data')
plot(D, m3(D),label='m3')
plot(D, m4(D),label='m4')
plot(D, m5(D),label='m5')
#plot(D, m6(D),label='m6')
plt.title('Emitted photon wavelength vs. quantum dot diameter')
plt.ylabel('$\lambda$ (nm)')
plt.xlabel('D (nm)')
plt.legend()
plt.savefig('CdSe_dotModel.png')




# ### The most complete model 5 (includes the kinetic energy of the free electron-hole pair) does not give good agreement for these CdSe dots.  Model 4 (simple spherical semiconductor dots) has the best reduced chi-square.