import tables
import os
from pathlib import Path
from astroquery.vizier import Vizier
import astropy.units as u
from importlib import resources
import astropy.coordinates as coords
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pystellibs import Munari,BaSeL,Kurucz,Tlusty
import pyphot
import math
import warnings
from pyphot import (unit, Filter)
import corner
import seaborn as sns
import extinction
import emcee
import pickle
import pyvo as vo
from matplotlib.ticker import FormatStrFormatter
from astropy.coordinates import SkyCoord
from importlib.resources import files
from .bijectors import *
mpl.rcParams["text.usetex"]=True
mpl.rcParams['font.family'] = 'serif'
Rs=696340000
kpc=3.08567758*10**21

directory = files("Iris.filters")
def mag_to_flux_pyphot(filtr,mag, err,AB):
    if err < 0: err = 0.2
    if AB:
        F_nu = pow(10.0, -0.4*mag) * filtr.AB_zero_Jy.magnitude # Jy
    else:
        F_nu = pow(10.0, -0.4*mag) * filtr.Vega_zero_Jy.magnitude # Jy
    #print(_filtr, F_nu)
    lam = filtr.leff.to("um").magnitude # um
    nu = 3.0e14 / lam # Hz
    #nuF_nu = nu*F_nu * 1.0e-26 # W/m^2
    F_lam = 3.0e-12 * F_nu / (lam*lam) # W/m^2/um
    lamF_lam = lam * F_lam # W/m^2
    frac_error = 0.4 * np.log(10.0) * err
    return lam, F_nu, F_lam, lamF_lam, frac_error

def to_temp(a,g):
    if (g==2.0):
        return (4.21-3.444)*a+3.444
    else:
        return (4.6-3.31)*a+3.31
def get_Vista(name):
    wave=np.loadtxt(directory.joinpath("VISTA_Filters_at80K_forETC_"+name+".dat"), unpack=True, usecols=[0], dtype=float)*10*unit["AA"]
    trans=np.loadtxt(directory.joinpath("VISTA_Filters_at80K_forETC_"+name+".dat"), unpack=True, usecols=[1], dtype=float)/100
    return Filter(wave,trans,name="Vista_"+name,unit='Angstrom')

def get_xmm(name):
    wave=np.loadtxt(directory.joinpath("XMM_OM."+name+".dat"), unpack=True, usecols=[0], dtype=float)*unit["AA"]
    trans=np.loadtxt(directory.joinpath("XMM_OM."+name+".dat"), unpack=True, usecols=[1], dtype=float)/100
    return Filter(wave,trans,name="XMMOM_"+name,unit='Angstrom')


def get_Denis(name):
    wave=np.loadtxt(directory.joinpath("DENIS_DENIS."+name+".dat"), unpack=True, usecols=[0], dtype=float)*unit["AA"]
    trans=np.loadtxt(directory.joinpath("DENIS_DENIS."+name+".dat"), unpack=True, usecols=[1], dtype=float)
    return Filter(wave,trans,name="DENIS_"+name,unit='Angstrom')

def get_IRSF(name):
    wave=np.loadtxt(directory.joinpath("IRSF_Sirius.{}.dat".format(name)), unpack=True, usecols=[0], dtype=float)*unit["AA"]
    trans=np.loadtxt(directory.joinpath("IRSF_Sirius.{}.dat".format(name)), unpack=True, usecols=[1], dtype=float)
    return Filter(wave,trans,name=name+"_IRSF",unit='Angstrom')

con={
    "J_2MASS":"2MASS_J",
    "H_2MASS":"2MASS_H",
    "K_2MASS":"2MASS_Ks",
    "J":"GROUND_BESSELL_J",
    "H":"GROUND_BESSELL_H",
    "K":"GROUND_BESSELL_K",
    "3.6":"SPITZER_IRAC_36",
    "4.5":"SPITZER_IRAC_45",
    "5.8":"SPITZER_IRAC_58",
    "8.0":"SPITZER_IRAC_80",
    "I":"GROUND_COUSINS_I",
    "V":"GROUND_JOHNSON_V",
    "g_SM":"SkyMapper_g",
    "r_SM":"SkyMapper_r",
    "i_SM":"SkyMapper_i",
    "z_SM":"SkyMapper_z",
    "u_SM":"SkyMapper_u",
    "v_SM":"SkyMapper_v",
    "B":"GROUND_JOHNSON_B",
    "U":"GROUND_JOHNSON_U",
    "G_Gaia":"GaiaDR2_G",
    "RP_Gaia":"GaiaDR2_RP",
    "BP_Gaia":"GaiaDR2_BP",
    "W1":"WISE_RSR_W1",
    "W2":"WISE_RSR_W2",
    "W3":"WISE_RSR_W3",
    "W4":"WISE_RSR_W4",
    "R":"GROUND_COUSINS_R",
    "g_PS1":"PS1_g",
    "i_PS1":"PS1_i",
    "z_PS1":"PS1_z",
    "r_PS1":"PS1_r",
    "T_TESS":"TESS",
    "FUV":"GALEX_FUV",
    "NUV":"GALEX_NUV"
}
AB=["NUV","FUV","i_PS1","z_PS1","g_PS1","r_PS1","g_SM","r_SM","i_SM","u_SM","v_SM","z_SM","UVM2","UVW1","UVW2"]
OTHER={}
OTHER["K_VISTA"]=get_Vista("Ks")
OTHER["Z_VISTA"]=get_Vista("Z")
OTHER["Y_VISTA"]=get_Vista("Y")
OTHER["J_VISTA"]=get_Vista("J")
OTHER["UVM2"]=get_xmm("UVM2")
OTHER["UVW2"]=get_xmm("UVW2")
OTHER["UVW1"]=get_xmm("UVW1")
OTHER["DENIS_I"]=get_Denis("I")
OTHER["DENIS_J"]=get_Denis("J")
OTHER["DENIS_Ks"]=get_Denis("Ks")
OTHER["H_IRSF"]=get_IRSF("H")
OTHER["Ks_IRSF"]=get_IRSF("Ks")
OTHER["J_IRSF"]=get_IRSF("J")
bijector_list_sig=[Sigmoid(low=3.31,high=4.6),Sigmoid(low=-3,high=5),Exp()]
bijector_list_double=[Sigmoid(low=3.31,high=4.6),Sigmoid(low=-3,high=5),Sigmoid(low=3.444,high=4.21),Sigmoid(low=-3,high=5),Exp()]


class Star:

    def __init__(self,name,ra,dec,catalog=None,d=None,d_err=None,paralax=None,paralax_err=None,E_B_V=None,Z=0.013):
        self.name=name
        self.ra=ra
        self.dec=dec
        self.plx=paralax
        self.e_plx=paralax_err
        self.d=d
        self.d_err=d_err
        self.filters=[]
        self.filters=[]
        self.ampl=[]
        self.err=[]
        self.zerop=[]
        self.dis_norm=1
        self.use_parallax=False
        self.catalogs=catalog
        self.EBV=E_B_V
        self.Z=Z
        self.par_single=np.zeros(3)
        self.par_single_container=None
        self.par_double=np.zeros(5)
        self.par_double_container=None
        self.lib_stell=BaSeL()
        self.lib_phot=pyphot.get_library()
        
        """
        DATA DOWNLOAD UTILS
        #########################################################################################################################
        """


    def get_SMdr2(self,num=0):
        if self.catalogs==None:
            print("No catalog found!")
            return
        service_sm = vo.dal.SCSService("https://skymapper.anu.edu.au/sm-cone/public/query?")
        v=SkyCoord(ra=self.ra,dec=self.dec,unit="deg")
        file = service_sm.search(pos=v, sr=0.3/(60*60))
        name="II/358/smss"
        if len(file)>0:
            print("found {} files in SM DR2".format(len(file)))
            for i in range(len(self.catalogs[name][0])//2):
                if str(file[num][self.catalogs[name][0][2*i]])=="nan" or str(file[num][self.catalogs[name][0][2*i+1]])=="nan":
                    pass
                else:
                    if (self.catalogs[name][1])[i] not in self.filters:
                        self.filters.append((self.catalogs[name][1])[i])
                        self.ampl.append(float(file[num][self.catalogs[name][0][2*i]]))
                        self.err.append(float(file[num][self.catalogs[name][0][2*i+1]]))
        else:
            print("no files found")

    def get_photo(self,name,num=0):
        if self.catalogs==None:
            print("No catalog found!")
            return
        v=Vizier(columns=self.catalogs[name][0])
        result = v.query_region(coords.SkyCoord(ra=self.ra, dec=self.dec,
                                            unit=(u.deg, u.deg),
                                            frame='icrs'),
                                            catalog=name,
                                            radius=self.catalogs[name][2]*u.arcsec)
        print(result)
        try:
            file=result[0]
            for i in range(len(self.catalogs[name][0])//2):
                if str(file[num][2*i])=="--" or str(file[num][2*i+1])=="--":
                    pass
                else:
                    if (self.catalogs[name][1])[i] not in self.filters:
                        self.filters.append((self.catalogs[name][1])[i])
                        self.ampl.append(float(file[num][2*i]))
                        self.err.append(float(file[num][2*i+1]))
        except IndexError:
            print("Object "+self.name+" not found in catalog "+name) 

    def delete(self,name):
        id=self.filters==name
        print(id)
        if sum(id)>0:
            self.filters=self.filters[np.logical_not(id)]
            self.err=self.err[np.logical_not(id)]
            self.ampl=self.ampl[np.logical_not(id)]
        else:
            print("no filter")
    
    def delete_id(self,id):
        temp=np.array([i==id for i in range(len(self.ampl))])
        self.ampl=self.ampl[np.logical_not(temp)]
        self.filters=self.filters[np.logical_not(temp)]
        self.err=self.err[np.logical_not(temp)]

    def get_all(self,get_SMDR2=True):
        if self.catalogs==None:
            print("No catalog found!")
            return
        if get_SMDR2:
            self.get_SMdr2()
        for name in self.catalogs:
            self.get_photo(name)
        if len(self.ampl)<10:
            warnings.warn("less then 10 points",Warning)


    def prepare_data(self):
        self.ampl=np.array(self.ampl)
        self.err=np.array(self.err)
        self.filters=list(map(lambda x: con[x] if x in con else x,self.filters))
        self.filters=np.array(self.filters)
        self.zerop=[]
        
        for i in self.filters:
            try:
                f=self.lib_phot[i]
            except tables.exceptions.NoSuchNodeError:
                f=OTHER[i]
            if i in AB:
                self.zerop.append(f.AB_zero_mag)
            else:
                self.zerop.append(f.Vega_zero_mag)
        self.fil_obj=[]
        for i in range(len(self.ampl)):
            try:
                self.fil_obj.append(self.lib_phot[self.filters[i]])
            except tables.exceptions.NoSuchNodeError:
                self.fil_obj.append(OTHER[self.filters[i]])
    
    def get_parallax(self):
        v=Vizier(columns=["Plx","e_Plx"])
        result = v.query_region(coords.SkyCoord(ra=self.ra, dec=self.dec,
                                            unit=(u.deg, u.deg),
                                            frame='icrs'),
                                            catalog="I/355 ",
                                            radius=0.3*u.arcsec)
        try:
            file=result[0]
            plx=file[0][0]
            e_plx=file[0][1]
            print("Parallax: {:.4f} mas".format(plx))
            print("Parallax error: {:.4f} mas".format(e_plx))
            if plx<3*e_plx:
                print("Parallax not statisticly significnt")
                self.plx=None
                self.e_plx=None
            else:
                self.plx=plx
                self.e_plx=e_plx
                self.use_parallax=True
                
        except IndexError:
            print("No parallax found for object")
            self.plx=None
            self.e_plx=None
    def set_EBV(self,ebv):
        self.EBV=ebv
        self.ext=extinction.fitzpatrick99(np.array(self.lib_stell.wavelength),3.1*self.EBV)
        print("E(B-V) = ",self.EBV)
    
    """
    MCMC ROUTINES
    ###########################################################################
    """

    @biject(bijector_list_sig)
    def get_log_prob_simple(self,i):
        logT,logL,d_p=i
        if self.use_parallax:
            d=1/d_p
        else:
            d=d_p
        pred=np.zeros(len(self.ampl))
        stell=self.lib_stell.generate_stellar_spectrum(logT,4.,logL,self.Z)/(4*math.pi*d**2*kpc**2)
        if self.EBV!=None:
            stell=np.power(10,-0.4*self.ext)*stell
        for i in range(len(self.ampl)):
            val=(self.fil_obj[i].get_flux(np.array(self.lib_stell.wavelength)*unit['AA'],np.array(stell)*unit['flam']))
            try:
                pred[i]=-2.5*np.log10(val.value)-self.zerop[i]
            except AttributeError:
                pred[i]=-2.5*np.log10(val)-self.zerop[i]
        return_val= -np.sum((pred-self.ampl)**2/(2*self.err**2))
        if self.use_parallax:
            return return_val-(d_p-self.plx)**2/(2*self.e_plx**2)
        else:
            return return_val-(d-self.d)**2/(2*self.d_err**2)

    @biject(bijector_list_double)
    def get_log_prob_double(self,i):
        logT1,logL1,logT2,logL2,d_p=i
        if self.use_parallax:
            d=1/d_p
        else:
            d=d_p
        pred=np.zeros(len(self.ampl))
        stell=(self.lib_stell.generate_stellar_spectrum(logT1,4.0,logL1,self.Z)+self.lib_stell.generate_stellar_spectrum(logT2,2.0,logL2,self.Z))/(4*math.pi*d**2*kpc**2)
        if self.EBV!=None:
            stell=np.power(10,-0.4*self.ext)*stell
        for i in range(len(self.ampl)):
            val=(self.fil_obj[i].get_flux(np.array(self.lib_stell.wavelength)*unit['AA'],np.array(stell)*unit['flam']))
            try:
                pred[i]=-2.5*np.log10(val.value)-self.zerop[i]
            except AttributeError:
                pred[i]=-2.5*np.log10(val)-self.zerop[i]
        return_val= -np.sum((pred-self.ampl)**2/(2*self.err**2))
        if self.use_parallax:
            return return_val-(d_p-self.plx)**2/(2*self.e_plx**2)
        else:
            return return_val-(d-self.d)**2/(2*self.d_err**2)


    def get_bic_double(self):
        logT_1=self.par_double[0]
        logT_2=self.par_double[2]
        logL_1=self.par_double[1]  
        logL_2=self.par_double[3]
        if self.use_parallax:
            stell=(self.lib_stell.generate_stellar_spectrum(logT_1,4.0,logL_1,self.Z)+self.lib_stell.generate_stellar_spectrum(logT_2,2.0,logL_2,self.Z))*self.par_double[4]**2/(4*math.pi*kpc**2)
        else:
            stell=(self.lib_stell.generate_stellar_spectrum(logT_1,4.0,logL_1,self.Z)+self.lib_stell.generate_stellar_spectrum(logT_2,2.0,logL_2,self.Z))/(4*math.pi*self.par_double[4]**2*kpc**2)
        if self.EBV!=None:
            stell=np.power(10,-0.4*self.ext)*stell
        pred=np.zeros(len(self.ampl))
        for i in range(len(self.ampl)):
            val=(self.fil_obj[i].get_flux(np.array(self.lib_stell.wavelength)*unit['AA'],np.array(stell)*unit['flam']))
            try:
                pred[i]=-2.5*np.log10(val.value)-self.zerop[i]
            except AttributeError:
                pred[i]=-2.5*np.log10(val)-self.zerop[i]
        chi2=np.sum((pred-self.ampl)**2/(self.err**2))
        bic=chi2+5*np.log(len(self.ampl))
        print("chi2: ",chi2)
        print("BIC: ",bic)
        self.bicd=bic
    def get_bic_simple(self):
        if self.use_parallax:
            stell=self.lib_stell.generate_stellar_spectrum(self.par_single[0],4.,self.par_single[1],self.Z)*self.par_single[2]**2/(4*math.pi*kpc**2)
        else:
            stell=self.lib_stell.generate_stellar_spectrum(self.par_single[0],4.,self.par_single[1],self.Z)/(4*math.pi*self.par_single[2]**2*kpc**2)
        if self.EBV!=None:
            stell=np.power(10,-0.4*self.ext)*stell
        pred=np.zeros(len(self.ampl))
        for i in range(len(self.ampl)):
            val=(self.fil_obj[i].get_flux(np.array(self.lib_stell.wavelength)*unit['AA'],np.array(stell)*unit['flam']))
            try:
                pred[i]=-2.5*np.log10(val.value)-self.zerop[i]
            except AttributeError:
                pred[i]=-2.5*np.log10(val)-self.zerop[i]
        chi2=np.sum((pred-self.ampl)**2/(self.err**2))
        bic=chi2+3*np.log(len(self.ampl))
        print("chi2: ",chi2)
        print("BIC: ",bic)
        self.bic_simple=bic

    def run_chain_double(self,num_step,num_burn,n,progress=True,use_simple_res=True):
        sampler = emcee.EnsembleSampler(
            n, 5, self.get_log_prob_double
            )
        
        start=np.repeat(np.array([[self.par_single[0],self.par_single[1],0,0,self.d]]),n,axis=0)
        start[:,2]=to_temp(np.random.rand(n),2.0)
        start[:,3]=np.random.rand(n)*8-3.
        if not use_simple_res:
            start[:,0]=to_temp(np.random.rand(n),4.0)
            start[:,1]=np.random.rand(n)*8-3.
        else:
            start[:,0]+=np.random.randn(n)*0.01
            start[:,1]+=np.random.randn(n)*0.01
        if self.use_parallax:
            start[:,4]=np.random.randn(n)*self.e_plx+self.plx
        else:
            start[:,4]=np.random.randn(n)*self.d_err+self.d
        print("starting conditions: ",start)
        sampler.run_mcmc(transform(start,bijector_list_double), num_step+num_burn, progress=progress)
        temp=untransform(sampler.get_chain(flat=True,discard=num_burn),bijector_list_double)
        logT1_samples,logL1_samples,logT2_samples,logL2_samples,d_samples=temp.T
        print("acceptence ratio:", np.mean(sampler.acceptance_fraction))
        self.par_double=[np.median(logT1_samples),np.median(logL1_samples),np.median(logT2_samples),np.median(logL2_samples),np.median(d_samples)]
        self.par_double_container=[logT1_samples,logL1_samples,logT2_samples,logL2_samples,d_samples]
        ###
        self.get_bic_double()
        self.log_prob_chainp=sampler.get_log_prob(flat=True,discard=num_burn)

    
    def run_chain_simple(self,num_step,num_burn,n,progress=True,T=None):
        sampler = emcee.EnsembleSampler(
            n, 3, self.get_log_prob_simple
            )
        start=np.zeros([n,3])
        if T is not None:
            T=0 if T<0 else T
            start_temp=np.log10(T)
            print("starting temprature", T,start_temp)
            start[:,0]=np.random.randn(1,n)*0.01+start_temp
        else:
            print("temprature estimate not found")
            start[:,0]=to_temp(np.random.rand(n),4.0)
        start[:,1]=np.random.rand(1,n)*8-3
        if self.use_parallax:
            start[:,2]=np.random.randn(n)*self.e_plx+self.plx
        else:
            start[:,2]=np.random.randn(n)*self.d_err+self.d
        print("starting conditions:", start)
        start_tf=transform(start,bijector_list_sig)
        sampler.run_mcmc(start_tf, num_step+num_burn, progress=progress)
        print("acceptance ratio",np.mean(sampler.acceptance_fraction))
        states=untransform(sampler.get_chain(flat=True,discard=num_burn),bijector_list_sig)
        logT_samples,logL_samples,val_samples=states.T
        self.par_single=[np.median(logT_samples),np.median(logL_samples),np.median(val_samples)]
        self.par_single_container=[logT_samples,logL_samples,val_samples]
        self.get_bic_simple()
        print("parameters:",self.par_single)
        self.log_prob_chain=sampler.get_log_prob(flat=True,discard=num_burn)

    def rerun_chain_double(self,num_step,num_burn,n,progress=True):
        sampler = emcee.EnsembleSampler(
        n, 5, self.get_log_prob_double
        )
        start=np.repeat(np.array([[self.par_single[0],self.par_single[1],0,0,0]]),n,axis=0)
        #start=np.repeat(np.array([[0,0,self.par_single[0],self.par_single[1],0]]),n,axis=0)
        start[:,0]=np.random.randn(n)*0.001+start[:,0]
        start[:,1]=np.random.randn(n)*0.001+start[:,1]
        start[:,2]=to_temp(np.random.rand(n),2.0)
        start[:,3]=np.random.rand(n)*8-3
        #start[:,0]=to_temp(np.random.rand(n),4.0)
        #start[:,1]=np.random.rand(n)*8-3
        if self.use_parallax:
            start[:,4]=np.random.randn(n)*self.e_plx+self.plx
        else:
            start[:,4]=np.random.randn(n)*self.d_err+self.d
        print("starting conditions:")
        print(start)
        sampler.run_mcmc(transform(start,bijector_list_double), num_step+num_burn, progress=progress)
        print("initial acceptence ratio",np.mean(sampler.acceptance_fraction))
        temp=untransform(sampler.get_chain(flat=True,discard=num_burn),bijector_list_double)
        log_probs=sampler.get_log_prob(flat=True,discard=num_burn)
        log_probs_tranc=np.unique(log_probs)
        temp_tran=np.unique(temp,axis=0)
        idx = (-log_probs_tranc).argsort()[:n]
        new_start=temp_tran[idx]+0.01 * np.random.randn(n, 5)
        print("new starting conditions:")
        print(new_start)
        sampler.reset()
        sampler.run_mcmc(transform(new_start,bijector_list_double), num_step, progress=progress)
        print("acceptence ratio",np.mean(sampler.acceptance_fraction))
        temp=untransform(sampler.get_chain(flat=True),bijector_list_double)
        logT1_samples,logL1_samples,logT2_samples,logL2_samples,v_samples=temp.T
        
        self.par_double=[np.median(logT1_samples),np.median(logL1_samples),np.median(logT2_samples),np.median(logL2_samples),np.median(v_samples)]
        self.par_double_container=[logT1_samples,logL1_samples,logT2_samples,logL2_samples,v_samples]
        ###
        self.get_bic_double()
        self.log_prob_chainp=sampler.get_log_prob(flat=True,discard=num_burn)

    def plot_measurments(self,ax,plot_fwhm=True):
        N = len(self.ampl)

        lam, lamF_lam, logerr = [], [], []
        lamFlam,lamFlam_err = [], []
        fwhm=[]
        for i in range(N):
            try:
                if self.filters[i] in AB:
                    AB_if=True
                else:
                    AB_if=False
                _lam, _F_nu, _F_lam, _lamF_lam, _frac_error = mag_to_flux_pyphot(self.fil_obj[i], self.ampl[i], self.err[i],AB_if)
                if self.filters[i] == 'H70' or self.filters[i] == 'H160' or self.filters[i] == 'H250' or self.filters[i] == 'H350':
                    warnings.warn('Herschel: check if magnitudes are in AB system!')
            except ValueError:
                warnings.warn('Unrecognized filter %s... skipping...'%self.filters[i])
                continue
            lam.append(_lam)
            lamF_lam.append(np.log10(_lamF_lam))
            logerr.append(_frac_error/np.log(10.0))
            lamFlam.append(_lamF_lam)
            lamFlam_err.append(_lamF_lam*_frac_error)
            try:
                fwhm_dim=self.fil_obj[i].fwhm.magnitude/2
            except AttributeError:
                fwhm_dim=self.fil_obj[i].fwhm/2
            if str(self.fil_obj[i].wavelength.units)=="angstrom":
                fwhm.append(fwhm_dim/10**4)
            elif str(self.fil_obj[i].wavelength.units)=="nanometer":
                fwhm.append(fwhm_dim/10**3)
            elif str(self.fil_obj[i].wavelength.units)=="micrometer":
                fwhm.append(fwhm_dim)
            else:
                print("error, unknown unit",self.fil_obj[i].wavelength.units)
                raise ValueError
        lamF_lam=np.array(lamF_lam)+7-4
        ax.set_ylim(min(lamF_lam)-0.5,max(lamF_lam)+2)
        ax.set_xlim(0.1,10)
        #for i in range(len(fwhm)):
        #    print(fwhm[i],self.filters[i])
        if len(lam)>0:
            if plot_fwhm:
                ax.errorbar(x=lam, y=lamF_lam, yerr=logerr,xerr=fwhm, fmt='o', mfc='navy', color='navy', ms=4, elinewidth=1.5,  capsize=2,label="measurements")
            else:
                ax.errorbar(x=lam, y=lamF_lam, yerr=logerr, fmt='o', mfc='navy', color='navy', ms=4, elinewidth=1.5,  capsize=2,label="measurements")
    
    def plot_dist_simple(self,n_l=2.2,bi=20,FWHM=False):
        plt.close()
        fig = plt.figure(figsize=(6,4))
        ax = plt.axes()
        self.plot_measurments(ax,FWHM)
        if self.use_parallax:
            flux=self.lib_stell.generate_stellar_spectrum(self.par_single[0],4.0,self.par_single[1],self.Z)*self.par_single[2]**2/(4*math.pi*kpc**2)
        else:
            flux=self.lib_stell.generate_stellar_spectrum(self.par_single[0],4.0,self.par_single[1],self.Z)/(4*math.pi*self.par_single[2]**2*kpc**2)
        if self.EBV!=None:
            flux=np.power(10,-0.4*self.ext)*flux
        r_sample=self.lib_stell.get_radius(self.par_single[1],self.par_single_container[0])
        if self.use_parallax:
            d_samples=1/self.par_single_container[2]
        else:
            d_samples=self.par_single_container[2]
        param=[10**self.par_single[0],10**np.quantile(self.par_single_container[0],0.84)-10**self.par_single[0],10**self.par_single[0]-10**np.quantile(self.par_single_container[0],0.16),
        np.median(d_samples),np.median(r_sample)/Rs,np.quantile(r_sample/Rs,0.84)-np.median(r_sample/Rs),np.median(r_sample)/Rs-np.quantile(r_sample/Rs,0.16),self.Z
        ]
        ax.plot(np.array(self.lib_stell.wavelength)/10**4,np.log10(np.array(flux)*np.array(self.lib_stell.wavelength)),
        label=r"$T={0:.0f}^{{+{1:.0f}}}_{{-{2:.0f}}}$ K, $ d={3:.2f}$ kpc, $R={4:.2f}^{{+{5:.2f}}}_{{-{6:.2f}}}$ $r_{{\odot}}$, $Z={7:.3f}$".format(*param),color="orange")
        ax.set_xscale('log')
        plt.legend()
        ax.set_xlabel(r'$\lambda$ [$\mu\textrm{m}$]')
        ax.set_ylabel(r'log $\lambda F_{\lambda}$ (erg cm$^{-2}$s$^{-1}$)')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.title(self.name)
        plt.figtext(0.25,0.05,"BIC: {0:.1f}".format(self.bic_simple))
        plt.tight_layout()
        plt.savefig(self.name+"_simple_emcee.png",dpi=500)

        plt.close()
        data=np.array([[self.par_single_container[0][i],self.par_single_container[1][i],self.par_single_container[2][i]] for i in range(len(self.par_single_container[0]))])
        print(len(data))
        upper=[(n_l+2)/2*np.quantile(self.par_single_container[0],0.84)-np.quantile(self.par_single_container[0],0.16)*n_l/2,
        (n_l+2)/2*np.quantile(self.par_single_container[1],0.84)-np.quantile(self.par_single_container[1],0.16)*n_l/2,
        (n_l+2)/2*np.quantile(self.par_single_container[2],0.84)-np.quantile(self.par_single_container[2],0.16)*n_l/2]
        lower=[(n_l+2)/2*np.quantile(self.par_single_container[0],0.16)-np.quantile(self.par_single_container[0],0.84)*n_l/2,
        (n_l+2)/2*np.quantile(self.par_single_container[1],0.16)-np.quantile(self.par_single_container[1],0.84)*n_l/2,
        (n_l+2)/2*np.quantile(self.par_single_container[2],0.16)-np.quantile(self.par_single_container[2],0.84)*n_l/2]
        upper[0]=min(upper[0],4.6)#+0.01
        lower[0]=max(lower[0],3.31)#-0.01
        upper[1]=upper[1]+0.01
        lower[1]=lower[1]-0.01
        if self.use_parallax:
            labels=[
            r"log $T$",
            r"log $L$",
            r"$\pi$",
            ]
        else:
            labels=[
            r"log $T$",
            r"log $L$",
            r"$d$",
            ]
        figure = corner.corner(
            data,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            title_fmt=".3f",
            bins=[int(bi/(upper[0]-lower[0])*(max(self.par_single_container[0])-min(self.par_single_container[0]))),
            int(bi/(upper[1]-lower[1])*(max(self.par_single_container[1])-min(self.par_single_container[1]))),
            int(bi/(upper[2]-lower[2])*(max(self.par_single_container[2])-min(self.par_single_container[2])))]
        )
        
        axes = np.array(figure.axes).reshape((3, 3))
        for i in range(3):
            for j in range(i+1):
                ax=axes[i,j]
                ax.set_xlim(lower[j],upper[j])
                if j<i:
                    ax.set_ylim(lower[i],upper[i])
        
        figure.suptitle(self.name)
        plt.savefig(self.name+"_simple_corner_emcee.png",dpi=500)


    def plot_dist_double(self,n_l=2.2,bi=20,FWHM=False):
        plt.close()
        fig = plt.figure(figsize=(6,4))
        ax=plt.axes()
        self.plot_measurments(ax,FWHM)
        r1_sample=self.lib_stell.get_radius(self.par_double_container[1],self.par_double_container[0])
        r2_sample=self.lib_stell.get_radius(self.par_double_container[3],self.par_double_container[2])
        if self.use_parallax:
            d_samples=1/self.par_double_container[4]
        else:
            d_samples=self.par_double_container[4]
        param1=[10**self.par_double[0],10**np.quantile(self.par_double_container[0],0.84)-10**self.par_double[0],10**self.par_double[0]-10**np.quantile(self.par_double_container[0],0.16),
        np.median(r1_sample)/Rs,np.quantile(r1_sample/Rs,0.84)-np.median(r1_sample/Rs),np.median(r1_sample)/Rs-np.quantile(r1_sample/Rs,0.16)
        ]
        param2=[10**self.par_double[2],10**np.quantile(self.par_double_container[2],0.84)-10**self.par_double[2],pow(10,self.par_double[2])-pow(10,np.quantile(self.par_double_container[2],0.16)),
        np.median(r2_sample)/Rs,np.quantile(r2_sample/Rs,0.84)-np.median(r2_sample/Rs),np.median(r2_sample)/Rs-np.quantile(r2_sample/Rs,0.16)
        ]
        print(param1,param2)
        flux1=self.lib_stell.generate_stellar_spectrum(self.par_double[0],4,self.par_double[1],self.Z)/(4*math.pi*np.median(d_samples)**2*kpc**2)
        flux2=self.lib_stell.generate_stellar_spectrum(self.par_double[2],3.5,self.par_double[3],self.Z)/(4*math.pi*np.median(d_samples)**2*kpc**2)
        if self.EBV!=None:
            flux1=np.power(10,-0.4*self.ext)*flux1
            flux2=np.power(10,-0.4*self.ext)*flux2
        ax.plot(np.array(self.lib_stell.wavelength)/10**4,np.log10(np.array(flux2)*np.array(self.lib_stell.wavelength)),
        label=r"$T={0:.0f}^{{+{1:.0f}}}_{{-{2:.0f}}}$ K, $R={3:.2f}^{{+{4:.2f}}}_{{-{5:.2f}}}$ $R_{{\odot}}$".format(*param2),color="red")
        ax.plot(np.array(self.lib_stell.wavelength)/10**4,np.log10(np.array(flux1)*np.array(self.lib_stell.wavelength)),
        label=r"$T={0:.0f}^{{+{1:.0f}}}_{{-{2:.0f}}}$ K, $R={3:.2f}^{{+{4:.2f}}}_{{-{5:.2f}}}$ $R_{{\odot}}$".format(*param1),color="yellow")
        ax.plot(np.array(self.lib_stell.wavelength)/10**4,np.log10(np.array(flux1+flux2)*np.array(self.lib_stell.wavelength)),label="combined",color="grey")
        
        ax.set_xscale('log')
        plt.legend()
        flux=flux1+flux2
        ax.set_xlabel(r'$\lambda$ [$\mu\textrm{m}$]')
        ax.set_ylabel(r'log $\lambda F_{\lambda}$ (erg cm$^{-2}$s$^{-1}$)')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.title(self.name)
        plt.figtext(0.25,0.05,"BIC: {0:.1f}".format(self.bicd))
        plt.figtext(0.75,0.05,"$d={0:.1f}$ kpc".format(np.median(d_samples)))
        plt.tight_layout()
        plt.savefig(self.name+"_double_emcee.png",dpi=500)
        print("parameters:",self.par_double)
        plt.close()
        data=np.array([[self.par_double_container[0][i],self.par_double_container[1][i],self.par_double_container[2][i],self.par_double_container[3][i],self.par_double_container[4][i]] for i in range(len(self.par_double_container[0]))])
        upper=[(n_l+2)/2*np.quantile(self.par_double_container[0],0.84)-np.quantile(self.par_double_container[0],0.16)*n_l/2,(n_l+2)/2*np.quantile(self.par_double_container[1],0.84)-np.quantile(self.par_double_container[1],0.16)*n_l/2,(n_l+2)/2*np.quantile(self.par_double_container[2],0.84)-np.quantile(self.par_double_container[2],0.16)*n_l/2,
        (n_l+2)/2*np.quantile(self.par_double_container[3],0.84)-np.quantile(self.par_double_container[3],0.16)*n_l/2,(n_l+2)/2*np.quantile(self.par_double_container[4],0.84)-np.quantile(self.par_double_container[4],0.16)*n_l/2]
        lower=[(n_l+2)/2*np.quantile(self.par_double_container[0],0.16)-np.quantile(self.par_double_container[0],0.84)*n_l/2,(n_l+2)/2*np.quantile(self.par_double_container[1],0.16)-np.quantile(self.par_double_container[1],0.84)*n_l/2,(n_l+2)/2*np.quantile(self.par_double_container[2],0.16)-np.quantile(self.par_double_container[2],0.84)*n_l/2,
        (n_l+2)/2*np.quantile(self.par_double_container[3],0.16)-np.quantile(self.par_double_container[3],0.84)*n_l/2,(n_l+2)/2*np.quantile(self.par_double_container[4],0.16)-np.quantile(self.par_double_container[4],0.84)*n_l/2]
        upper[0]=min(upper[0],4.6)
        upper[1]=min(upper[1],5)
        upper[2]=min(upper[2],4.21)
        upper[3]=min(upper[3],5)
        lower[0]=max(lower[0],3.31)
        lower[1]=max(lower[1],-3)
        lower[2]=max(lower[2],3.444)
        lower[3]=max(lower[3],-3)
        if self.use_parallax:
            labels=[
            r"log $T_1$",
            r"log $L_1$",
            r"log $T_2$",
            r"log $L_2$",
            r"$\pi$",
            ]
        else:
            labels=[
            r"log $T_1$",
            r"log $L_1$",
            r"log $T_2$",
            r"log $L_2$",
            r"$d$",
            ]
        figure = corner.corner(
            data,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            title_fmt=".3f",
            bins=[int(bi/(upper[0]-lower[0])*(max(self.par_double_container[0])-min(self.par_double_container[0]))),
            int(bi/(upper[1]-lower[1])*(max(self.par_double_container[1])-min(self.par_double_container[1]))),
            int(bi/(upper[2]-lower[2])*(max(self.par_double_container[2])-min(self.par_double_container[2]))),
            int(bi/(upper[3]-lower[3])*(max(self.par_double_container[3])-min(self.par_double_container[3]))),
            int(bi/(upper[4]-lower[4])*(max(self.par_double_container[4])-min(self.par_double_container[4])))]
            )
            
        axes = np.array(figure.axes).reshape((5, 5))
        for i in range(5):
            for j in range(i+1):
                ax=axes[i,j]
                ax.set_xlim(lower[j],upper[j])
                if j<i:
                    ax.set_ylim(lower[i],upper[i])
        figure.suptitle(self.name)
        plt.savefig(self.name+"_double_corner_emcee.png",dpi=500)

    def plot_dist_simple_density(self,num_samples,n_l=2.2,bi=20,FWHM=False):
        plt.close()
        fig = plt.figure(figsize=(6,4))
        ax = plt.axes()
        self.plot_measurments(ax,FWHM)
        flux=np.zeros([num_samples,1221])
        wave=np.zeros([num_samples,1221])
        id_list=np.random.randint(low=0, high=len(self.par_single_container[0]), size=num_samples)
        if self.use_parallax:
            for i in range(num_samples):
                id=id_list[i]
                flux[i]=self.lib_stell.generate_stellar_spectrum(self.par_single_container[0][id],4.0,self.par_single_container[1][id],self.Z)*self.par_single_container[2][id]**2/(4*math.pi*kpc**2)
                wave[i]=np.array(self.lib_stell.wavelength)
        else:
            for i in range(num_samples):
                id=id_list[i]
                flux[i]=self.lib_stell.generate_stellar_spectrum(self.par_single_container[0][id],4.0,self.par_single_container[1][id],self.Z)/(4*math.pi*kpc**2*self.par_single_container[2][id]**2)
                wave[i]=np.array(self.lib_stell.wavelength)
        if self.EBV!=None:
            flux=np.power(10,-0.4*self.ext)*flux
        r_sample=self.lib_stell.get_radius(self.par_single[1],self.par_single_container[0])
        if self.use_parallax:
            d_samples=1/self.par_single_container[2]
        else:
            d_samples=self.par_single_container[2]
        param=[10**self.par_single[0],10**np.quantile(self.par_single_container[0],0.84)-10**self.par_single[0],10**self.par_single[0]-10**np.quantile(self.par_single_container[0],0.16),
        np.median(d_samples),np.median(r_sample)/Rs,np.quantile(r_sample/Rs,0.84)-np.median(r_sample/Rs),np.median(r_sample)/Rs-np.quantile(r_sample/Rs,0.16),self.Z
        ]
        sns.lineplot(x=wave.flatten()/10**4,y=np.log10(flux.flatten()*wave.flatten()),ax=ax,
        label=r"$T={0:.0f}^{{+{1:.0f}}}_{{-{2:.0f}}}$ K, $ d={3:.2f}$ kpc, $R={4:.2f}^{{+{5:.2f}}}_{{-{6:.2f}}}$ $r_{{\odot}}$, $Z={7:.3f}$".format(*param),color="orange")
        ax.set_xscale('log')
        plt.legend()
        ax.set_xlabel(r'$\lambda$ [$\mu\textrm{m}$]')
        ax.set_ylabel(r'log $\lambda F_{\lambda}$ (erg cm$^{-2}$s$^{-1}$)')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.title(self.name)
        plt.figtext(0.25,0.05,"BIC: {0:.1f}".format(self.bic_simple))
        plt.tight_layout()
        plt.savefig(self.name+"_simple_emcee.png",dpi=500)
        print("parameters:",self.par_single)
        print("errors:",self.par_single_container)
        plt.close()
        data=np.array([[self.par_single_container[0][i],self.par_single_container[1][i],self.par_single_container[2][i]] for i in range(len(self.par_single_container[0]))])
        print(len(data))
        upper=[(n_l+2)/2*np.quantile(self.par_single_container[0],0.84)-np.quantile(self.par_single_container[0],0.16)*n_l/2,
        (n_l+2)/2*np.quantile(self.par_single_container[1],0.84)-np.quantile(self.par_single_container[1],0.16)*n_l/2,
        (n_l+2)/2*np.quantile(self.par_single_container[2],0.84)-np.quantile(self.par_single_container[2],0.16)*n_l/2]
        lower=[(n_l+2)/2*np.quantile(self.par_single_container[0],0.16)-np.quantile(self.par_single_container[0],0.84)*n_l/2,
        (n_l+2)/2*np.quantile(self.par_single_container[1],0.16)-np.quantile(self.par_single_container[1],0.84)*n_l/2,
        (n_l+2)/2*np.quantile(self.par_single_container[2],0.16)-np.quantile(self.par_single_container[2],0.84)*n_l/2]
        upper[0]=min(upper[0],4.6)#+0.01
        lower[0]=max(lower[0],3.31)#-0.01
        upper[1]=upper[1]+0.01
        lower[1]=lower[1]-0.01
        if self.use_parallax:
            labels=[
            r"log $T$",
            r"log $L$",
            r"$\pi$",
            ]
        else:
            labels=[
            r"log $T$",
            r"log $L$",
            r"$d$",
            ]
        figure = corner.corner(
            data,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            title_fmt=".3f",
            bins=[int(bi/(upper[0]-lower[0])*(max(self.par_single_container[0])-min(self.par_single_container[0]))),
            int(bi/(upper[1]-lower[1])*(max(self.par_single_container[1])-min(self.par_single_container[1]))),
            int(bi/(upper[2]-lower[2])*(max(self.par_single_container[2])-min(self.par_single_container[2])))]
        )
        axes = np.array(figure.axes).reshape((3, 3))
        for i in range(3):
            for j in range(i+1):
                ax=axes[i,j]
                ax.set_xlim(lower[j],upper[j])
                if j<i:
                    ax.set_ylim(lower[i],upper[i])
        
        figure.suptitle(self.name)
        plt.savefig(self.name+"_simple_corner_emcee.png",dpi=500)

    def plot_dist_double_density(self,num_samples,n_l=2,bi=8,FWHM=False):
        plt.close()
        fig = plt.figure(figsize=(6,4))
        ax=plt.axes()
        self.plot_measurments(ax,FWHM)
        r1_sample=self.lib_stell.get_radius(self.par_double_container[1],self.par_double_container[0])
        r2_sample=self.lib_stell.get_radius(self.par_double_container[3],self.par_double_container[2])
        if self.use_parallax:
            d_samples=1/self.par_double_container[4]
        else:
            d_samples=self.par_double_container[4]
        param1=[10**self.par_double[0],10**np.quantile(self.par_double_container[0],0.84)-10**self.par_double[0],10**self.par_double[0]-10**np.quantile(self.par_double_container[0],0.16),
        np.median(r1_sample)/Rs,np.quantile(r1_sample/Rs,0.84)-np.median(r1_sample/Rs),np.median(r1_sample)/Rs-np.quantile(r1_sample/Rs,0.16)
        ]
        param2=[10**self.par_double[2],10**np.quantile(self.par_double_container[2],0.84)-10**self.par_double[2],pow(10,self.par_double[2])-pow(10,np.quantile(self.par_double_container[2],0.16)),
        np.median(r2_sample)/Rs,np.quantile(r2_sample/Rs,0.84)-np.median(r2_sample/Rs),np.median(r2_sample)/Rs-np.quantile(r2_sample/Rs,0.16)
        ]
        print(param1,param2)
        flux1=np.zeros([num_samples,1221])
        flux2=np.zeros([num_samples,1221])
        wave=np.zeros([num_samples,1221])
        id_list=np.random.randint(low=0, high=len(self.par_single_container[0]), size=num_samples)
        if self.use_parallax:
            for i in range(num_samples):
                id=id_list[i]
                flux1[i]=self.lib_stell.generate_stellar_spectrum(self.par_double_container[0][id],4.0,self.par_double_container[1][id],self.Z)*self.par_double_container[4][id]**2/(4*math.pi*kpc**2)
                flux2[i]=self.lib_stell.generate_stellar_spectrum(self.par_double_container[2][id],4.0,self.par_double_container[3][id],self.Z)*self.par_double_container[4][id]**2/(4*math.pi*kpc**2)
                wave[i]=np.array(self.lib_stell.wavelength)
        else:
            for i in range(num_samples):
                id=id_list[i]
                flux1[i]=self.lib_stell.generate_stellar_spectrum(self.par_double_container[0][id],4.0,self.par_double_container[1][id],self.Z)/(4*math.pi*kpc**2*self.par_double_container[4][id]**2)
                flux2[i]=self.lib_stell.generate_stellar_spectrum(self.par_double_container[2][id],4.0,self.par_double_container[3][id],self.Z)/(4*math.pi*kpc**2*self.par_double_container[4][id]**2)
                wave[i]=np.array(self.lib_stell.wavelength)
        if self.EBV!=None:
            flux1=np.power(10,-0.4*self.ext)*flux1
            flux2=np.power(10,-0.4*self.ext)*flux2
        sns.lineplot(x=wave.flatten()/10**4,y=np.log10(flux1.flatten()*wave.flatten()),ax=ax,
        label=r"$T={0:.0f}^{{+{1:.0f}}}_{{-{2:.0f}}}$ K, $R={3:.2f}^{{+{4:.2f}}}_{{-{5:.2f}}}$ $R_{{\odot}}$".format(*param2),color="red")
        sns.lineplot(x=wave.flatten()/10**4,y=np.log10(flux2.flatten()*wave.flatten()),ax=ax,
        label=r"$T={0:.0f}^{{+{1:.0f}}}_{{-{2:.0f}}}$ K, $R={3:.2f}^{{+{4:.2f}}}_{{-{5:.2f}}}$ $R_{{\odot}}$".format(*param1),color="yellow")
        sns.lineplot(x=wave.flatten()/10**4,y=np.log10((flux1+flux2).flatten()*wave.flatten()),ax=ax,
        label="combined",color="grey")
        
        ax.set_xscale('log')
        plt.legend()
        flux=flux1+flux2
        ax.set_xlabel(r'$\lambda$ [$\mu\textrm{m}$]')
        ax.set_ylabel(r'log $\lambda F_{\lambda}$ (erg cm$^{-2}$s$^{-1}$)')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.title(self.name)
        plt.figtext(0.25,0.05,"BIC: {0:.1f}".format(self.bicd))
        plt.figtext(0.75,0.05,"$d={0:.1f}$ kpc".format(np.median(d_samples)))
        plt.tight_layout()
        plt.savefig(self.name+"_double_emcee.png",dpi=500)
        print("parameters:",self.par_double)
        #print("errors:",self.par_double_container)
        plt.close()
        data=np.array([[self.par_double_container[0][i],self.par_double_container[1][i],self.par_double_container[2][i],self.par_double_container[3][i],self.par_double_container[4][i]] for i in range(len(self.par_double_container[0]))])
        upper=[(n_l+2)/2*np.quantile(self.par_double_container[0],0.84)-np.quantile(self.par_double_container[0],0.16)*n_l/2,(n_l+2)/2*np.quantile(self.par_double_container[1],0.84)-np.quantile(self.par_double_container[1],0.16)*n_l/2,(n_l+2)/2*np.quantile(self.par_double_container[2],0.84)-np.quantile(self.par_double_container[2],0.16)*n_l/2,
        (n_l+2)/2*np.quantile(self.par_double_container[3],0.84)-np.quantile(self.par_double_container[3],0.16)*n_l/2,(n_l+2)/2*np.quantile(self.par_double_container[4],0.84)-np.quantile(self.par_double_container[4],0.16)*n_l/2]
        lower=[(n_l+2)/2*np.quantile(self.par_double_container[0],0.16)-np.quantile(self.par_double_container[0],0.84)*n_l/2,(n_l+2)/2*np.quantile(self.par_double_container[1],0.16)-np.quantile(self.par_double_container[1],0.84)*n_l/2,(n_l+2)/2*np.quantile(self.par_double_container[2],0.16)-np.quantile(self.par_double_container[2],0.84)*n_l/2,
        (n_l+2)/2*np.quantile(self.par_double_container[3],0.16)-np.quantile(self.par_double_container[3],0.84)*n_l/2,(n_l+2)/2*np.quantile(self.par_double_container[4],0.16)-np.quantile(self.par_double_container[4],0.84)*n_l/2]
        upper[0]=min(upper[0],4.6)
        upper[1]=min(upper[1],5)
        upper[2]=min(upper[2],4.21)
        upper[3]=min(upper[3],5)
        lower[0]=max(lower[0],3.31)
        lower[1]=max(lower[1],-3)
        lower[2]=max(lower[2],3.444)
        lower[3]=max(lower[3],-3)
        if self.use_parallax:
            labels=[
            r"log $T_1$",
            r"log $L_1$",
            r"log $T_2$",
            r"log $L_2$",
            r"$\pi$",
            ]
        else:
            labels=[
            r"log $T_1$",
            r"log $L_1$",
            r"log $T_2$",
            r"log $L_2$",
            r"$d$",
            ]
        figure = corner.corner(
            data,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            title_fmt=".3f",
            bins=[int(bi/(upper[0]-lower[0])*(max(self.par_double_container[0])-min(self.par_double_container[0]))),
            int(bi/(upper[1]-lower[1])*(max(self.par_double_container[1])-min(self.par_double_container[1]))),
            int(bi/(upper[2]-lower[2])*(max(self.par_double_container[2])-min(self.par_double_container[2]))),
            int(bi/(upper[3]-lower[3])*(max(self.par_double_container[3])-min(self.par_double_container[3]))),
            int(bi/(upper[4]-lower[4])*(max(self.par_double_container[4])-min(self.par_double_container[4])))]
            )
            
        axes = np.array(figure.axes).reshape((5, 5))
        for i in range(5):
            for j in range(i+1):
                ax=axes[i,j]
                ax.set_xlim(lower[j],upper[j])
                if j<i:
                    ax.set_ylim(lower[i],upper[i])
        figure.suptitle(self.name)
        plt.savefig(self.name+"_double_corner_emcee.png",dpi=500)