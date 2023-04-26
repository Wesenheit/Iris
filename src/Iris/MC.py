import tables
from pathlib import Path
from astroquery.vizier import Vizier
import astropy.units as u
import astropy.coordinates as coords
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pystellibs import Munari,BaSeL,Kurucz,Tlusty,Marcs,Elodie
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
import h5py
from typing import Optional,Tuple,List
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
    lam = filtr.leff.to("um").magnitude # um
    nu = 3.0e14 / lam # Hz
    F_lam = 3.0e-12 * F_nu / (lam*lam) # W/m^2/um
    lamF_lam = lam * F_lam # W/m^2
    frac_error = 0.4 * np.log(10.0) * err
    return lam, F_nu, F_lam, lamF_lam, frac_error



"""
functions to create filters from .dat files

"""
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

def get_SWIFT(name):
    wave=np.loadtxt(directory.joinpath("Swift_UVOT."+name+".dat"), unpack=True, usecols=[0], dtype=float)*unit["AA"]
    trans=np.loadtxt(directory.joinpath("Swift_UVOT."+name+".dat"), unpack=True, usecols=[1], dtype=float)/100
    return Filter(wave,trans,name="SWIFT_"+name,unit='Angstrom')

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
    "y_PS1":"PS1_y",
    "T_TESS":"TESS",
    "FUV":"GALEX_FUV",
    "NUV":"GALEX_NUV"
}
#AB=["NUV","FUV","i_PS1","z_PS1","g_PS1","r_PS1","g_SM","r_SM","i_SM","u_SM","v_SM","z_SM","UVM2","UVW1","UVW2","SDSS_g","SDSS_r","SDSS_i"]
AB=["GALEX_NUV","GALEX_FUV","PS1_g","PS1_i","PS1_z","PS1_r","PS1_y","SkyMapper_g","SkyMapper_r","SkyMapper_i","SkyMapper_z","SkyMapper_u","SkyMapper_v",
"UVM2","UVW1","UVW2","SDSS_g","SDSS_r","SDSS_i","SWIFT_UVM2","SWIFT_UVW2","SWIFT_UVW1"]
OTHER={}
OTHER["K_VISTA"]=get_Vista("Ks")
OTHER["Z_VISTA"]=get_Vista("Z")
OTHER["Y_VISTA"]=get_Vista("Y")
OTHER["J_VISTA"]=get_Vista("J")
OTHER["H_VISTA"]=get_Vista("H")
OTHER["UVM2"]=get_xmm("UVM2")
OTHER["UVW2"]=get_xmm("UVW2")
OTHER["UVW1"]=get_xmm("UVW1")
OTHER["SWIFT_UVM2"]=get_SWIFT("UVM2")
OTHER["SWIFT_UVW2"]=get_SWIFT("UVW2")
OTHER["SWIFT_UVW1"]=get_SWIFT("UVW1")
OTHER["DENIS_I"]=get_Denis("I")
OTHER["DENIS_J"]=get_Denis("J")
OTHER["DENIS_Ks"]=get_Denis("Ks")
OTHER["H_IRSF"]=get_IRSF("H")
OTHER["Ks_IRSF"]=get_IRSF("Ks")
OTHER["J_IRSF"]=get_IRSF("J")


class Star:
    """
    main Star class
    usage:
    -construct list of filters with coresponding magnitudes and errors
    -remember to pass distance or parallax!
    -call preproces data, this is crucial before sampling!
    -run chains and plot data
    """
    def __init__(self,name:str,ra:float,dec:float,catalog=None,d:Optional[float]=None,
                 d_err:Optional[float]=None,parallax:Optional[float]=None,parallax_err:Optional[float]=None,
                 E_B_V:Optional[float]=None,Z:Optional[float]=0.013):
        self.name=name                      #name
        self.ra=ra                          #right ascesion angle
        self.dec=dec                        #declination
        self.plx=parallax                   #parallax 
        self.e_plx=parallax_err             #parallax error
        self.d=d                            #distance
        self.d_err=d_err                    #error of distance
        self.filters=np.array([],dtype=str) #filters
        self.ampl=np.array([])              #magnitudes
        self.err=np.array([])               #errors
        self.zerop=[]                       #zero points of filters
        self.dis_norm=1                     #norm to search distance
        self.use_parallax=False             #use parallax to sample or not
        self.catalogs=catalog               #catalog in the specified format to search Vizier
        self.EBV=E_B_V                      #E(B-V), used to include extinction using 
        self.Z=Z                            #metalicity
        self.par_single=np.zeros(3)         #parameters of single sample
        self.par_single_container=None      #container for values
        self.par_double=np.zeros(5)         #parameters of double sample
        self.par_double_container=None      #container for values
        self.lib_stell=BaSeL()              #what library to use

        self.lib_phot=pyphot.get_library()
        self.gp=2
        self.g1=4
        self.g2=2
        """
        DATA DOWNLOAD UTILS
        #########################################################################################################################
        Iris recognizes two types of filters:
        -recognized by pyphot
            -in orginal name as in pyphot
            -mapped to orginal name using converting dictionary con
        -added manually
        """
    def to_temp(self,a,g,n=1000):
        logt_arr=np.linspace(0,5.5,n)
        g_arr=np.ones_like(logt_arr)*g
        arr=np.stack((logt_arr,g_arr))
        is_in=self.lib_stell.points_inside(arr.T)
        t_min=np.min(logt_arr[is_in])
        t_max=np.max(logt_arr[is_in])
        return t_min+(t_max-t_min)*a
    
    def get_boundaries(self,g,n=1000):
        logt_arr=np.linspace(0,5.5,n)
        g_arr=np.ones_like(logt_arr)*g
        arr=np.stack((logt_arr,g_arr))
        is_in=self.lib_stell.points_inside(arr.T)
        t_min=np.min(logt_arr[is_in])
        t_max=np.max(logt_arr[is_in])
        return t_min,t_max
    def get_SMdr2(self,num=0):
        """
        get data from SkyMapper DR2, unfortunately we cannot use Vizier :(
        """
        if self.catalogs==None:
            print("No catalog found!")
            return
        service_sm = vo.dal.SCSService("https://skymapper.anu.edu.au/sm-cone/public/query?")
        v=SkyCoord(ra=self.ra,dec=self.dec,unit="deg")
        file = service_sm.search(pos=v, sr=0.3/(60*60)*self.dis_norm)
        name="II/358/smss"
        if len(file)>0:
            print("found {} files in SM DR2".format(len(file)))
            for i in range(len(self.catalogs[name][0])//2):
                if str(file[num][self.catalogs[name][0][2*i]])=="nan":
                    print("Something missing in {}!".format(name))
                else:
                    err=0 if str(file[num][self.catalogs[name][0][2*i+1]])=="nan" else float(file[num][self.catalogs[name][0][2*i+1]])
                    name_new=con[(self.catalogs[name][1])[i]] if (self.catalogs[name][1])[i] in con else (self.catalogs[name][1])[i]
                    if name_new not in self.filters:
                        name_new=con[(self.catalogs[name][1])[i]] if (self.catalogs[name][1])[i] in con else (self.catalogs[name][1])[i]
                        self.filters=np.append(self.filters,name_new)
                        self.ampl=np.append(self.ampl,float(file[num][self.catalogs[name][0][2*i]]))
                        self.err=np.append(self.err,err)
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
                                            radius=self.catalogs[name][2]*u.arcsec*self.dis_norm)
        try:
            file=result[0]
            print(result)
            for i in range(len(self.catalogs[name][0])//2):
                if str(file[num][2*i])=="--":
                    print("Something missing in {}!".format(name))
                else:
                    err=0 if str(file[num][2*i+1])=="--" else float(file[num][2*i+1])
                    name_new=con[(self.catalogs[name][1])[i]] if (self.catalogs[name][1])[i] in con else (self.catalogs[name][1])[i]
                    if name_new not in self.filters:
                        self.filters=np.append(self.filters,name_new)
                        self.ampl=np.append(self.ampl,float(file[num][2*i]))
                        self.err=np.append(self.err,err)
        except IndexError:
            pass
            #print("Object "+self.name+" not found in catalog "+name) 

    def delete(self,name):
        """
        delete one filter
        """
        id=(self.filters==name)
        if sum(id)>0:
            self.filters=self.filters[np.logical_not(id)]
            self.err=self.err[np.logical_not(id)]
            self.ampl=self.ampl[np.logical_not(id)]
        else:
            print("no filter with given name")
    
    def delete_id(self,id):
        """
        delete filter with given id
        """
        temp=np.array([i==id for i in range(len(self.ampl))])
        self.ampl=self.ampl[np.logical_not(temp)]
        self.filters=self.filters[np.logical_not(temp)]
        self.err=self.err[np.logical_not(temp)]

    def get_all(self,get_SMDR2=True):
        """
        use provided catalogs to find data
        """
        if self.catalogs==None:
            print("No catalog found!")
            return
        if get_SMDR2:
            self.get_SMdr2()
        for name in self.catalogs:
            self.get_photo(name)
        if len(self.ampl)<10:
            warnings.warn("less then 10 points",Warning)


    def prepare_data(self,replace="max"):
        """
        prepare data before usage
        """
        names=[]
        for i,(err,name) in enumerate(zip(self.err,self.filters)):
            if err<=0:
                if replace=="max":
                    self.err[i]=np.max(self.err)
                    print("filter {} with no magnitude error! Setting to max".format(name))
                elif type(replace)==float:
                    self.err[i]=replace
                    print("filter {} with no magnitude error! Setting to {}".format(name,replace))
                else:
                    names.append(name)
                    print("filter {} with no magnitude error!".format(name))

        self.zerop=[]
        for n in names:
            self.delete(n)

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
    
    def get_parallax(self,show=True,ratio=3):
        """
        get parallax using Gaia DR3
        """
        v=Vizier(columns=["Plx","e_Plx"])
        result = v.query_region(coords.SkyCoord(ra=self.ra, dec=self.dec,
                                            unit=(u.deg, u.deg),
                                            frame='icrs'),
                                            catalog="I/355/gaiadr3",
                                            radius=0.3*u.arcsec*self.dis_norm)
        try:
            file=result[0]
            plx=file[0][0]
            e_plx=file[0][1]
            if show:
                print("Parallax: {:.4f} mas".format(plx))
                print("Parallax error: {:.4f} mas".format(e_plx))
            if plx<ratio*e_plx:
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

    def get_pos_gaia(self,show=True):
        """
        get position using Gaia DR3
        """
        v=Vizier(columns=["RA_ICRS","DE_ICRS"])
        result = v.query_region(coords.SkyCoord(ra=self.ra, dec=self.dec,
                                            unit=(u.deg, u.deg),
                                            frame='icrs'),
                                            catalog="I/355/gaiadr3 ",
                                            radius=0.3*u.arcsec*self.dis_norm)
        try:
            file=result[0]
            ra=file[0][0]
            dec=file[0][1]
            if show:
                print("RA: {:.4f} deg".format(ra))
                print("DEC: {:.4f} deg".format(dec))
            return ra,dec
                
        except IndexError:
            print("No gaia entry found for object")

    def get_radial_velocity(self,show=True):
        """
        get velocity using Gaia DR3
        """
        v=Vizier(columns=["RV","e_RV","o_RV"])
        result = v.query_region(coords.SkyCoord(ra=self.ra, dec=self.dec,
                                            unit=(u.deg, u.deg),
                                            frame='icrs'),
                                            catalog="I/355/gaiadr3 ",
                                            radius=0.3*u.arcsec*self.dis_norm)
        try:
            file=result[0]
            v=file[0][0]
            ev=file[0][1]
            N=file[0][2]
            if show:
                print("Vr: {:.4f} km/s".format(v))
                print("Vr error: {:.4f} km/s".format(ev))
                print("estimated using {} transits".format(N))
            return v,ev,N
                
        except IndexError:
            print("No gaia entry found for object")

    def set_EBV(self,ebv):
        """
        set E(B-V) using provided value
        """
        self.EBV=ebv
        self.ext=extinction.ccm89(np.array(self.lib_stell.wavelength).astype(np.double),3.1*self.EBV,3.1)
        print("E(B-V) = ",self.EBV)

    def add_obs(self,name,ampl,err):
        """
        manually add observation
        """
        self.filters.append(name)
        self.ampl.append(ampl)
        self.err.append(err)

    def get_EBV_TIC(self):
        """
        get E(B-V) using TIC
        """
        v=Vizier(columns=["E(B-V) ","s_E(B-V) "])
        result = v.query_region(coords.SkyCoord(ra=self.ra, dec=self.dec,
                                            unit=(u.deg, u.deg),
                                            frame='icrs'),
                                            catalog="IV/38/tic ",
                                            radius=0.3*u.arcsec)
        try:
            file=result[0]
            ebv=file[0][0]
            e_ebv=file[0][1]
            print("E(B-V): {:.4f} mas".format(ebv))
            print("E(B-V) err: {:.4f} mas".format(e_ebv))
            self.EBV=ebv
        except IndexError:
            print("No E(B-V) found for object")

    def list_filters(self):
        for i in range(len(self.filters)):
            print(self.filters[i],self.ampl[i],self.err[i])

    """
    MCMC ROUTINES
    ###########################################################################
    """

    def predict_single(self,list_filters):
        """
        Predict values for single model
        """
        pred=np.zeros(len(list_filters))
        if self.use_parallax:
            d=1/self.par_single[2]
        else:
            d=self.par_single[2]
        stell=self.lib_stell.generate_stellar_spectrum(self.par_single[0],self.gp,self.par_single[1],self.Z)/(4*math.pi*d**2*kpc**2)
        if self.EBV!=None:
            stell=np.power(10,-0.4*self.ext)*stell
        for i in range(len(list_filters)):
            try:
                val=(self.lib_phot[list_filters[i]].get_flux(np.array(self.lib_stell.wavelength)*unit['AA'],np.array(stell)*unit['flam']))
                if list_filters[i] in AB:
                    pred[i]=-2.5*np.log10(val.value)-self.lib_phot[list_filters[i]].AB_zero_mag
                else:
                    pred[i]=-2.5*np.log10(val.value)-self.lib_phot[list_filters[i]].Vega_zero_mag
            except AttributeError:
                val=(OTHER[list_filters[i]].get_flux(np.array(self.lib_stell.wavelength)*unit['AA'],np.array(stell)*unit['flam']))
                if list_filters[i] in AB:
                    pred[i]=-2.5*np.log10(val)-OTHER[list_filters[i]].AB_zero_mag
                else:
                    pred[i]=-2.5*np.log10(val)-OTHER[list_filters[i]].Vega_zero_mag
        return pred

    def predict_double(self,list_filters):
        """
        predict values for double star model
        """
        pred=np.zeros(len(list_filters))
        if self.use_parallax:
            d=1/self.par_double[4]
        else:
            d=self.par_double[4]
        stell=(self.lib_stell.generate_stellar_spectrum(self.par_double[0],self.g1,self.par_single[1],self.Z)+
                self.lib_stell.generate_stellar_spectrum(self.par_double[2],self.g2,self.par_double[3],self.Z))/(4*math.pi*d**2*kpc**2)
        if self.EBV!=None:
            stell=np.power(10,-0.4*self.ext)*stell
        for i in range(len(list_filters)):
            try:
                val=(self.lib_phot[list_filters[i]].get_flux(np.array(self.lib_stell.wavelength)*unit['AA'],np.array(stell)*unit['flam']))
                if list_filters[i] in AB:
                    pred[i]=-2.5*np.log10(val.value)-self.lib_phot[list_filters[i]].AB_zero_mag
                else:
                    pred[i]=-2.5*np.log10(val.value)-self.lib_phot[list_filters[i]].Vega_zero_mag
            except AttributeError:
                val=(OTHER[list_filters[i]].get_flux(np.array(self.lib_stell.wavelength)*unit['AA'],np.array(stell)*unit['flam']))
                if list_filters[i] in AB:
                    pred[i]=-2.5*np.log10(val)-OTHER[list_filters[i]].AB_zero_mag
                else:
                    pred[i]=-2.5*np.log10(val)-OTHER[list_filters[i]].Vega_zero_mag
        return pred

    def get_log_prob_simple(self,i):
        """
        log prob of observations giving single star model
        """
        logT,logL,d_p=i
        if self.use_parallax:
            d=1/d_p
        else:
            d=d_p
        pred=np.zeros(len(self.ampl))
        stell=self.lib_stell.generate_stellar_spectrum(logT,self.gp,logL,self.Z)/(4*math.pi*d**2*kpc**2)
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

    def get_log_prob_double(self,i):
        """
        log prob of observations given 2 star model
        """
        logT1,logL1,logT2,logL2,d_p=i
        if self.use_parallax:
            d=1/d_p
        else:
            d=d_p
        pred=np.zeros(len(self.ampl))
        stell=(self.lib_stell.generate_stellar_spectrum(logT1,self.g1,logL1,self.Z)+self.lib_stell.generate_stellar_spectrum(logT2,self.g2,logL2,self.Z))/(4*math.pi*d**2*kpc**2)
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
        """
        get bic score for double star model
        """
        logT_1=self.par_double[0]
        logT_2=self.par_double[2]
        logL_1=self.par_double[1]  
        logL_2=self.par_double[3]
        if self.use_parallax:
            stell=(self.lib_stell.generate_stellar_spectrum(logT_1,self.g1,logL_1,self.Z)+self.lib_stell.generate_stellar_spectrum(logT_2,self.g2,logL_2,self.Z))*self.par_double[4]**2/(4*math.pi*kpc**2)
        else:
            stell=(self.lib_stell.generate_stellar_spectrum(logT_1,self.g1,logL_1,self.Z)+self.lib_stell.generate_stellar_spectrum(logT_2,self.g2,logL_2,self.Z))/(4*math.pi*self.par_double[4]**2*kpc**2)
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
        """
        get bic value for double star model
        """
        if self.use_parallax:
            stell=self.lib_stell.generate_stellar_spectrum(self.par_single[0],self.gp,self.par_single[1],self.Z)*self.par_single[2]**2/(4*math.pi*kpc**2)
        else:
            stell=self.lib_stell.generate_stellar_spectrum(self.par_single[0],self.gp,self.par_single[1],self.Z)/(4*math.pi*self.par_single[2]**2*kpc**2)
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

    def run_chain_double(self,num_step:int,num_burn:int,n:int,
                         progress:Optional[bool]=True,use_simple_res:Optional[bool]=False):
        """
        run chain for double star model
        num_step - number of steps
        num_burn - number of burn-in steps
        n - number of walkers 
        progress - progres bar
        use_simple_res - whether to use values for single star model as starting point
        """
        start=np.repeat(np.array([[self.par_single[0],self.par_single[1],0,0,self.d]]),n,axis=0)
        start[:,2]=self.to_temp(np.random.rand(n),self.g2)
        start[:,3]=np.random.rand(n)*8-3.
        if not use_simple_res:
            start[:,0]=self.to_temp(np.random.rand(n),self.g1)
            start[:,1]=np.random.rand(n)*8-3.
        else:
            start[:,0]+=np.random.randn(n)*0.01
            start[:,1]+=np.random.randn(n)*0.01
        if self.use_parallax:
            start[:,4]=np.random.randn(n)*self.e_plx+self.plx
        else:
            start[:,4]=np.random.randn(n)*self.d_err+self.d
        print("starting conditions: ",start)
        logT1_low,logT1_high=self.get_boundaries(self.g1)
        logT2_low,logT2_high=self.get_boundaries(self.g2)
        bijector_list_double=[Sigmoid(logT1_low,logT1_high),Sigmoid(-3,5),Sigmoid(logT2_low,logT2_high),Sigmoid(-3,5),Exp()]

        sampler = emcee.EnsembleSampler(
            n, 5, (biject(bijector_list_double))(self.get_log_prob_double)
            )
        sampler.run_mcmc(transform(start,bijector_list_double), num_step+num_burn, progress=progress)
        temp=untransform(sampler.get_chain(flat=True,discard=num_burn),bijector_list_double)
        logT1_samples,logL1_samples,logT2_samples,logL2_samples,d_samples=temp.T
        print("acceptence ratio:", np.mean(sampler.acceptance_fraction))
        self.par_double=[np.median(logT1_samples),np.median(logL1_samples),np.median(logT2_samples),np.median(logL2_samples),np.median(d_samples)]
        self.par_double_container=temp.T
        ###
        self.get_bic_double()
        self.log_prob_chainp=sampler.get_log_prob(flat=True,discard=num_burn)

    def run_chain_double_start(self,num_step:int,num_burn:int,n:int,
                               parameters:np.array,progress:Optional[bool]=True): 
        """
        run chain for double star model
        num_step - number of steps
        num_burn - number of burn-in steps
        n - number of walkers 
        progress - progres bar
        parameters - starting parameters of chain
        """  
        if len(parameters.shape)>1:
            start=parameters
        else:
            start=np.repeat(np.array([parameters]),n,axis=0)+np.random.randn(n,5)*0.01
    
        logT1_low,logT1_high=self.get_boundaries(self.g1)
        logT2_low,logT2_high=self.get_boundaries(self.g2)
        bijector_list_double=[Sigmoid(logT1_low,logT1_high),Sigmoid(-3,5),Sigmoid(logT2_low,logT2_high),Sigmoid(-3,5),Exp()]

        sampler = emcee.EnsembleSampler(
            n, 5, (biject(bijector_list_double))(self.get_log_prob_double)
            )
        print("starting conditions: ",start)
        sampler.run_mcmc(transform(start,bijector_list_double), num_step+num_burn, progress=progress)
        temp=untransform(sampler.get_chain(flat=True,discard=num_burn),bijector_list_double)
        logT1_samples,logL1_samples,logT2_samples,logL2_samples,d_samples=temp.T
        print("acceptence ratio:", np.mean(sampler.acceptance_fraction))
        self.par_double=[np.median(logT1_samples),np.median(logL1_samples),np.median(logT2_samples),np.median(logL2_samples),np.median(d_samples)]
        self.par_double_container=temp.T
        ###
        self.get_bic_double()
        self.log_prob_chainp=sampler.get_log_prob(flat=True,discard=num_burn)

    
    def run_chain_simple(self,num_step,num_burn,n,progress=True,T=None):
        logT_low,logT_high=self.get_boundaries(self.gp)
        bijector_list_sig=[Sigmoid(logT_low,logT_high),Sigmoid(-3,5),Exp()]
        sampler = emcee.EnsembleSampler(
            n, 3, (biject(bijector_list_sig))(self.get_log_prob_simple)
            )

        start=np.zeros([n,3])
        if T is not None:
            T=0 if T<0 else T
            start_temp=np.log10(T)
            print("starting temprature", T,start_temp)
            start[:,0]=np.random.randn(1,n)*0.01+start_temp
        else:
            print("temprature estimate not found")
            start[:,0]=self.to_temp(np.random.rand(n),self.gp)
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
        self.par_single_container=states.T
        self.get_bic_simple()
        print("parameters:",self.par_single)
        self.log_prob_chain=sampler.get_log_prob(flat=True,discard=num_burn)

    def rerun_chain_simple(self,num_step,num_burn,n,progress=True,T=None,max_prob=False):
        logT_low,logT_high=self.get_boundaries(self.gp)
        bijector_list_sig=[Sigmoid(logT_low,logT_high),Sigmoid(-3,5),Exp()]
        sampler = emcee.EnsembleSampler(
            n, 3, (biject(bijector_list_sig))(self.get_log_prob_simple)
            )
        start=np.zeros([n,3])
        if T is not None:
            T=0 if T<0 else T
            start_temp=np.log10(T)
            print("starting temprature", T,start_temp)
            start[:,0]=np.random.randn(1,n)*0.01+start_temp
        else:
            print("temprature estimate not found")
            start[:,0]=self.to_temp(np.random.rand(n),self.gp)
        start[:,1]=np.random.rand(1,n)*8-3
        if self.use_parallax:
            start[:,2]=np.random.randn(n)*self.e_plx+self.plx
        else:
            start[:,2]=np.random.randn(n)*self.d_err+self.d
        print("starting conditions:", start)
        start_tf=transform(start,bijector_list_sig)
        sampler.run_mcmc(start_tf, num_step+num_burn, progress=progress)
        print("inital acceptance ratio",np.mean(sampler.acceptance_fraction))
        temp=untransform(sampler.get_chain(flat=True,discard=num_burn),bijector_list_sig)
        if max_prob:
            print("using maximum log prob to start")
            log_probs=sampler.get_log_prob(flat=True,discard=num_burn)
            log_probs_tranc=np.unique(log_probs)
            temp_tran=np.unique(temp,axis=0)
            idx = (-log_probs_tranc).argsort()[:n]
            new_start=temp_tran[idx]+0.01 * np.random.randn(n, 3)
        else:
            print("sampling new starting conditions")
            id = np.random.permutation(len(temp_tran))[:n]
            new_start = temp_tran[id] + np.random.randn(n,3) * 0.01
        print("new starting conditions:")
        print(new_start)
        sampler.reset()
        sampler.run_mcmc(transform(new_start,bijector_list_sig), num_step, progress=progress)
        print("acceptence ratio",np.mean(sampler.acceptance_fraction))
        states=untransform(sampler.get_chain(flat=True,discard=num_burn),bijector_list_sig)
        logT_samples,logL_samples,val_samples=states.T
        self.par_single=[np.median(logT_samples),np.median(logL_samples),np.median(val_samples)]
        self.par_single_container=states.T
        self.get_bic_simple()
        print("parameters:",self.par_single)
        self.log_prob_chain=sampler.get_log_prob(flat=True,discard=num_burn)


    def rerun_chain_double(self,num_step:int,num_burn:int,n:int,
                           progress:Optional[bool]=True,
                           max_prob:Optional[bool]=False,
                           use_simple_res:Optional[bool]=False):
        start=np.repeat(np.array([[self.par_single[0],self.par_single[1],0,0,0]]),n,axis=0)
        if use_simple_res:
            start[:,0]=np.random.randn(n)*0.001+start[:,0]
            start[:,1]=np.random.randn(n)*0.001+start[:,1]
        else:
            start[:,0]=self.to_temp(np.random.rand(n),self.g1)
            start[:,1]=np.random.rand(n)*8-3
        start[:,2]=self.to_temp(np.random.rand(n),self.g2)
        start[:,3]=np.random.rand(n)*8-3
        if self.use_parallax:
            start[:,4]=np.random.randn(n)*self.e_plx+self.plx
        else:
            start[:,4]=np.random.randn(n)*self.d_err+self.d
        logT1_low,logT1_high=self.get_boundaries(self.g1)
        logT2_low,logT2_high=self.get_boundaries(self.g2)
        bijector_list_double=[Sigmoid(logT1_low,logT1_high),Sigmoid(-3,5),Sigmoid(logT2_low,logT2_high),Sigmoid(-3,5),Exp()]

        sampler = emcee.EnsembleSampler(
            n, 5, (biject(bijector_list_double))(self.get_log_prob_double)
            )
        print("starting conditions:")
        print(start)
        sampler.run_mcmc(transform(start,bijector_list_double), num_step+num_burn, progress=progress)
        print("initial acceptence ratio",np.mean(sampler.acceptance_fraction))
        
        temp=untransform(sampler.get_chain(flat=True,discard=num_burn),bijector_list_double)
        print("values after first run: ",np.median(temp,axis=0))
        temp_tran = np.unique(temp,axis=0)
        if max_prob:
            log_probs = sampler.get_log_prob(flat=True,discard=num_burn)
            log_probs_tranc = np.unique(log_probs)
            idx = (-log_probs_tranc).argsort()[:n]
            new_start = temp_tran[idx]+0.01 * np.random.randn(n, 5)
        else:
            print("sampling new starting conditions")
            id = np.random.permutation(len(temp_tran))[:n]
            new_start = temp_tran[id] + np.random.randn(n,5) * 0.01

        print("new starting conditions:")
        print(new_start)
        sampler.reset()
        sampler.run_mcmc(transform(new_start,bijector_list_double), num_step, progress=progress)
        print("acceptence ratio",np.mean(sampler.acceptance_fraction))
        temp=untransform(sampler.get_chain(flat=True),bijector_list_double)
        logT1_samples,logL1_samples,logT2_samples,logL2_samples,v_samples=temp.T
        
        self.par_double=[np.median(logT1_samples),np.median(logL1_samples),np.median(logT2_samples),np.median(logL2_samples),np.median(v_samples)]
        self.par_double_container=temp.T
        ###
        self.get_bic_double()
        self.log_prob_chainp=sampler.get_log_prob(flat=True,discard=num_burn)

    def plot_measurments(self,ax,plot_fwhm=False):
        N = len(self.ampl)

        lam, lamF_lam, logerr = [], [], []
        lamFlam,lamFlam_err = [], []
        fwhm=[]
        for i in range(N):
            if self.filters[i] in AB:
                AB_if=True
            else:
                AB_if=False
            _lam, _F_nu, _F_lam, _lamF_lam, _frac_error = mag_to_flux_pyphot(self.fil_obj[i], self.ampl[i], self.err[i],AB_if)

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
        if self.EBV is not None:
            plt.figtext(0.65,0.05,r"$E(B-V)={0:.3f}$".format(self.EBV))
        if len(lam)>0:
            if plot_fwhm:
                ax.errorbar(x=lam, y=lamF_lam, yerr=lamFlam_err,xerr=fwhm, fmt='o', mfc='navy', color='navy', ms=4, elinewidth=1.5,  capsize=2,label="measurements")
            else:
                ax.errorbar(x=lam, y=lamF_lam, yerr=lamFlam_err, fmt='o', mfc='navy', color='navy', ms=4, elinewidth=1.5,  capsize=2,label="measurements")
    
    def plot_dist_simple(self,FWHM=False,ax=None):
        plt.close()
        if ax is None:
            fig = plt.figure(figsize=(6,4))
            ax = plt.axes()
        self.plot_measurments(ax,FWHM)
        if self.use_parallax:
            flux=self.lib_stell.generate_stellar_spectrum(self.par_single[0],self.gp,self.par_single[1],self.Z)*self.par_single[2]**2/(4*math.pi*kpc**2)
        else:
            flux=self.lib_stell.generate_stellar_spectrum(self.par_single[0],self.gp,self.par_single[1],self.Z)/(4*math.pi*self.par_single[2]**2*kpc**2)
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
        label=r"$T={0:.0f}^{{+{1:.0f}}}_{{-{2:.0f}}}$ K, $R={4:.2f}^{{+{5:.2f}}}_{{-{6:.2f}}}$ $R_{{\odot}}$, $Z={7:.3f}$".format(*param),color="orange")
        ax.set_xscale('log')
        plt.legend()
        ax.set_xlabel(r'$\lambda$ [$\mu\textrm{m}$]')
        ax.set_ylabel(r'log $\lambda F_{\lambda}$ (erg cm$^{-2}$s$^{-1}$)')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.title(self.name)
        plt.figtext(0.2,0.05,"BIC: {0:.1f}".format(self.bic_simple))
        if self.EBV is None:
            high=0.05
        else:
            high=0.1
        plt.figtext(0.65,high,"$d={0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$ kpc".format(np.median(d_samples),
                np.quantile(d_samples,0.84)-np.median(d_samples),
                np.median(d_samples)-np.quantile(d_samples,0.16)
            ))
        plt.tight_layout()
        plt.savefig(self.name+"_simple_emcee.png",dpi=500)

    def plot_corner_simple(self,n_l=2.2,bi=20,):
        data=self.par_single_container.T
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


    def plot_dist_double(self,FWHM=False,ax=None):
        if ax is None:
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
        flux1=self.lib_stell.generate_stellar_spectrum(self.par_double[0],self.g1,self.par_double[1],self.Z)/(4*math.pi*np.median(d_samples)**2*kpc**2)
        flux2=self.lib_stell.generate_stellar_spectrum(self.par_double[2],self.g2,self.par_double[3],self.Z)/(4*math.pi*np.median(d_samples)**2*kpc**2)
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
        plt.figtext(0.2,0.05,"BIC: {0:.1f}".format(self.bicd))
        if self.EBV is None:
            high=0.05
        else:
            high=0.1
        plt.figtext(0.65,high,"$d={0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$ kpc".format(np.median(d_samples),
                np.quantile(d_samples,0.84)-np.median(d_samples),
                np.median(d_samples)-np.quantile(d_samples,0.16)
            ))
        plt.tight_layout()
        plt.savefig(self.name+"_double_emcee.png",dpi=500)
        print("parameters:",self.par_double)

    def plot_corner_double(self,n_l=2.2,bi=20):
        data=self.par_double_container.T
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
                flux[i]=self.lib_stell.generate_stellar_spectrum(self.par_single_container[0][id],self.gp,self.par_single_container[1][id],self.Z)*self.par_single_container[2][id]**2/(4*math.pi*kpc**2)
                wave[i]=np.array(self.lib_stell.wavelength)
        else:
            for i in range(num_samples):
                id=id_list[i]
                flux[i]=self.lib_stell.generate_stellar_spectrum(self.par_single_container[0][id],self.gp,self.par_single_container[1][id],self.Z)/(4*math.pi*kpc**2*self.par_single_container[2][id]**2)
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
        label=r"$T={0:.0f}^{{+{1:.0f}}}_{{-{2:.0f}}}$ K, $R={4:.2f}^{{+{5:.2f}}}_{{-{6:.2f}}}$ $R_{{\odot}}$, $Z={7:.3f}$".format(*param),color="orange")
        ax.set_xscale('log')
        plt.legend()
        ax.set_xlabel(r'$\lambda$ [$\mu\textrm{m}$]')
        ax.set_ylabel(r'log $\lambda F_{\lambda}$ (erg cm$^{-2}$s$^{-1}$)')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.title(self.name)
        plt.figtext(0.2,0.05,"BIC: {0:.1f}".format(self.bic_simple))
        if self.EBV is None:
            high=0.05
        else:
            high=0.1
        plt.figtext(0.65,high,"$d={0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$ kpc".format(np.median(d_samples),
                np.quantile(d_samples,0.84)-np.median(d_samples),
                np.median(d_samples)-np.quantile(d_samples,0.16)
            ))
        plt.tight_layout()
        plt.savefig(self.name+"_simple_emcee.png",dpi=500)
        print("parameters:",self.par_single)
        #print("errors:",self.par_single_container)
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
        id_list=np.random.randint(low=0, high=len(self.par_double_container[0]), size=num_samples)
        if self.use_parallax:
            for i in range(num_samples):
                id=id_list[i]
                flux1[i]=self.lib_stell.generate_stellar_spectrum(self.par_double_container[0][id],self.g1,self.par_double_container[1][id],self.Z)*self.par_double_container[4][id]**2/(4*math.pi*kpc**2)
                flux2[i]=self.lib_stell.generate_stellar_spectrum(self.par_double_container[2][id],self.g2,self.par_double_container[3][id],self.Z)*self.par_double_container[4][id]**2/(4*math.pi*kpc**2)
                wave[i]=np.array(self.lib_stell.wavelength)
        else:
            for i in range(num_samples):
                id=id_list[i]
                flux1[i]=self.lib_stell.generate_stellar_spectrum(self.par_double_container[0][id],self.g1,self.par_double_container[1][id],self.Z)/(4*math.pi*kpc**2*self.par_double_container[4][id]**2)
                flux2[i]=self.lib_stell.generate_stellar_spectrum(self.par_double_container[2][id],self.g2,self.par_double_container[3][id],self.Z)/(4*math.pi*kpc**2*self.par_double_container[4][id]**2)
                wave[i]=np.array(self.lib_stell.wavelength)
        if self.EBV!=None:
            flux1=np.power(10,-0.4*self.ext)*flux1
            flux2=np.power(10,-0.4*self.ext)*flux2
        sns.lineplot(x=wave.flatten()/10**4,y=np.log10(flux1.flatten()*wave.flatten()),ax=ax,
        label=r"$T={0:.0f}^{{+{1:.0f}}}_{{-{2:.0f}}}$ K, $R={3:.2f}^{{+{4:.2f}}}_{{-{5:.2f}}}$ $R_{{\odot}}$".format(*param1),color="red")
        sns.lineplot(x=wave.flatten()/10**4,y=np.log10(flux2.flatten()*wave.flatten()),ax=ax,
        label=r"$T={0:.0f}^{{+{1:.0f}}}_{{-{2:.0f}}}$ K, $R={3:.2f}^{{+{4:.2f}}}_{{-{5:.2f}}}$ $R_{{\odot}}$".format(*param2),color="yellow")
        sns.lineplot(x=wave.flatten()/10**4,y=np.log10((flux1+flux2).flatten()*wave.flatten()),ax=ax,
        label="combined",color="grey")
        
        ax.set_xscale('log')
        plt.legend()
        flux=flux1+flux2
        ax.set_xlabel(r'$\lambda$ [$\mu\textrm{m}$]')
        ax.set_ylabel(r'log $\lambda F_{\lambda}$ (erg cm$^{-2}$s$^{-1}$)')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.title(self.name)
        plt.figtext(0.2,0.05,"BIC: {0:.1f}".format(self.bicd))
        if self.EBV is None:
            high=0.05
        else:
            high=0.1
        plt.figtext(0.65,high,"$d={0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$ kpc".format(np.median(d_samples),
                np.quantile(d_samples,0.84)-np.median(d_samples),
                np.median(d_samples)-np.quantile(d_samples,0.16)
            ))
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

    def save(self,name=None):
        if name is None:
            name=self.name+".h5"
        file = h5py.File(name, 'w')
        if self.par_double_container is not None:
            file.create_dataset("double",data=self.par_double)
            file.create_dataset("double_container",data=self.par_double_container)
        if self.par_single_container is not None:
            file.create_dataset("single",data=self.par_single)
            file.create_dataset("single_container",data=self.par_single_container)
        ascii_type = h5py.string_dtype('ascii', 30)
        file.create_dataset("bands",data=self.filters.astype(ascii_type))
        file.create_dataset("mag",data=self.ampl)
        file.create_dataset("err",data=self.err)
        if self.d is not None:
            file.create_dataset("d",data=np.array([self.d,self.d_err]))
        if self.plx is not None:
            self.create_dataset("plx",data=np.array([self.plx,self.e_plx]))
        file.create_dataset("gp",data=(self.gp,))
        file.create_dataset("g1",data=(self.g1,))
        file.create_dataset("g2",data=(self.g2,))
        file.create_dataset("Z",data=(self.Z,))
        file.close()

    def load(self,name=None):
        file = h5py.File(name, 'r')
        if "single" in file.keys():
            self.par_single=np.array(file.get("single"))
            self.par_single_container=np.array(file.get("single_container"))
        if "double" in file.keys():
            self.par_double=np.array(file.get("double"))
            self.par_double_container=np.array(file.get("double_container"))
        if "bands" in file.keys():
            self.filters=np.array(file.get("bands"))
        if "mag" in file.keys():
            self.ampl=np.array(file.get("mag"))
        if "err" in file.keys():
            self.err=np.array(file.get("err"))
        self.gp=file.get("gp")
        self.g1=file.get("g1")
        self.g2=file.get("g2")
        self.Z=file.get("Z")
        file.close()