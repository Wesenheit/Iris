import tables
from pathlib import Path
from astroquery.mast import Catalogs
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


def get_UKIDSS(name):
    wave=np.loadtxt(directory.joinpath("UKIRT_UKIDSS."+name+".dat"), unpack=True, usecols=[0], dtype=float)*unit["AA"]
    trans=np.loadtxt(directory.joinpath("UKIRT_UKIDSS."+name+".dat"), unpack=True, usecols=[1], dtype=float)/100
    return Filter(wave,trans,name="UKIDSS_"+name,unit='Angstrom')


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
        self.Z1=Z
        self.Z2=Z
        self.par_single=None                #parameters of single sample
        self.par_single_container=None      #container for values
        self.par_double=None                #parameters of double sample
        self.par_double_container=None      #container for values
        self.lib_stell=BaSeL()              #what library to use
        self.fluxes_ = None
        self.flux_err = None
        self.err_estimate_flag = np.array([],dtype=bool)
        self.error_estimate = None

        self.lib_phot=pyphot.get_library()
        self.gp=4
        self.g1=4
        self.g2=2
        self.AB=["GALEX_NUV","GALEX_FUV","PS1_g","PS1_i","PS1_z","PS1_r","PS1_y","SkyMapper_g","SkyMapper_r","SkyMapper_i","SkyMapper_z","SkyMapper_u","SkyMapper_v",
        "UVM2","UVW1","UVW2","SDSS_g","SDSS_r","SDSS_i","SDSS_u","SDSS_z","SWIFT_UVM2","SWIFT_UVW2","SWIFT_UVW1"]
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
        OTHER["H_UKIDSS"]=get_UKIDSS("H")
        OTHER["J_UKIDSS"]=get_UKIDSS("J")
        OTHER["K_UKIDSS"]=get_UKIDSS("K")
        OTHER["Y_UKIDSS"]=get_UKIDSS("Y")
        OTHER["Z_UKIDSS"]=get_UKIDSS("Z")
        self.OTHER=OTHER

        self.con={
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
    
    def add_error_estimate(self,filter,error):
        try:
            f=self.lib_phot[filter]
        except tables.exceptions.NoSuchNodeError:
            f=self.OTHER[filter]
        self.error_estimate = (f,error)
    def add_error_estimate_double(self,filter,error):
        try:
            f=self.lib_phot[filter]
        except tables.exceptions.NoSuchNodeError:
            f=self.OTHER[filter]
        self.error_estimate_double = (f,error)

    def get_GALEX(self,num=0):
        results = Catalogs.query_object(str(self.ra)+" "+str(self.dec), catalog="Galex",radius=u.arcsec*2*self.dis_norm)
        if len(results)>0:
            print("Found {} files in GALEX MAST".format(len(results)))
            try:
                nuv = float(results["nuv_mag"][num])
            except:
                nuv = 0
            try:
                n_err = float(results["nuv_magerr"][num])
            except:
                n_err = 0
            try:
                fuv = float(results["fuv_mag"][num])
            except:
                fuv = 0
            try:
                f_err = float(results["fuv_magerr"][num])
            except:
                f_err = 0
            if nuv>0:
                self.add_obs("GALEX_NUV",nuv,n_err,flag=True)
            if fuv>0:
                self.add_obs("GALEX_FUV",fuv,f_err,flag=True)
        else:
            print("No GALEX data in MAST!")
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
                    name_new=self.con[(self.catalogs[name][1])[i]] if (self.catalogs[name][1])[i] in self.con else (self.catalogs[name][1])[i]
                    if name_new not in self.filters:
                        name_new=self.con[(self.catalogs[name][1])[i]] if (self.catalogs[name][1])[i] in self.con else (self.catalogs[name][1])[i]
                        self.filters=np.append(self.filters,name_new)
                        self.ampl=np.append(self.ampl,float(file[num][self.catalogs[name][0][2*i]]))
                        self.err=np.append(self.err,err)
                        self.err_estimate_flag = np.append(self.err_estimate_flag,False)
        else:
            print("no files found")

    def get_photo(self,name,num=0,**kwargs):
        if self.catalogs==None:
            print("No catalog found!")
            return
        v=Vizier(columns=self.catalogs[name][0],**kwargs)
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
                    name_new=self.con[(self.catalogs[name][1])[i]] if (self.catalogs[name][1])[i] in self.con else (self.catalogs[name][1])[i]
                    if name_new not in self.filters:
                        self.filters=np.append(self.filters,name_new)
                        self.ampl=np.append(self.ampl,float(file[num][2*i]))
                        self.err=np.append(self.err,err)
                        self.err_estimate_flag = np.append(self.err_estimate_flag,self.catalogs[name][3][i])
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
            self.err_estimate_flag=self.err_estimate_flag[np.logical_not(id)]
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
        self.err_estimate_flag=self.err_estimate_flag[np.logical_not(temp)]

    def get_all(self,get_SMDR2=True,get_galex_mast=1,**kwargs):
        """
        use provided catalogs to find data
        """
        if self.catalogs==None:
            print("No catalog found!")
            return
        if get_SMDR2:
            self.get_SMdr2()
        if get_galex_mast>0:
            self.get_GALEX(get_galex_mast-1)
        for name in self.catalogs:
            self.get_photo(name,**kwargs)
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
                f=self.OTHER[i]
            if i in self.AB:
                self.zerop.append(f.AB_zero_flux.magnitude)
            else:
                self.zerop.append(f.Vega_zero_flux.magnitude)
        self.fil_obj=[]
        for i in range(len(self.ampl)):
            try:
                self.fil_obj.append(self.lib_phot[self.filters[i]])
            except tables.exceptions.NoSuchNodeError:
                self.fil_obj.append(self.OTHER[self.filters[i]])
        self.zerop = np.array(self.zerop)
        self.fluxes_ = np.power(10,-0.4*self.ampl)
        self.err_flux = self.fluxes_ * np.log(10)*0.4*self.err
    
    def get_parallax(self,show=True,ratio=3,**kwargs):
        """
        get parallax using Gaia DR3
        """
        v=Vizier(columns=["Plx","e_Plx"],**kwargs)
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

    def get_pos_gaia(self,show=True,**kwargs):
        """
        get position using Gaia DR3
        """
        v=Vizier(columns=["RA_ICRS","DE_ICRS"],**kwargs)
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

    def add_obs(self,name,ampl,err,flag=True):
        """
        manually add observation
        """
        self.filters=np.append(self.filters,self.con[name] if name in self.con else name)
        self.ampl=np.append(self.ampl,ampl)
        self.err=np.append(self.err,err)
        self.err_estimate_flag = np.append(self.err_estimate_flag,flag)

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

    def list_filters(self,show_flag=False):
        for i in range(len(self.filters)):
            if show_flag:
                print("{} {:.5g} {:.2g} {}".format(self.filters[i],self.ampl[i],self.err[i],self.err_estimate_flag[i]))
            else:
                print("{} {:.5g} {:.2g}".format(self.filters[i],self.ampl[i],self.err[i]))

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
                if list_filters[i] in self.AB:
                    pred[i]=-2.5*np.log10(val.value)-self.lib_phot[list_filters[i]].AB_zero_mag
                else:
                    pred[i]=-2.5*np.log10(val.value)-self.lib_phot[list_filters[i]].Vega_zero_mag
            except AttributeError:
                val=(self.OTHER[list_filters[i]].get_flux(np.array(self.lib_stell.wavelength)*unit['AA'],np.array(stell)*unit['flam']))
                if list_filters[i] in self.AB:
                    pred[i]=-2.5*np.log10(val)-self.OTHER[list_filters[i]].AB_zero_mag
                else:
                    pred[i]=-2.5*np.log10(val)-self.OTHER[list_filters[i]].Vega_zero_mag
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
        stell=(self.lib_stell.generate_stellar_spectrum(self.par_double[0],self.g1,self.par_double[1],self.Z)+
                self.lib_stell.generate_stellar_spectrum(self.par_double[2],self.g2,self.par_double[3],self.Z))/(4*math.pi*d**2*kpc**2)
        if self.EBV!=None:
            stell=np.power(10,-0.4*self.ext)*stell
        for i in range(len(list_filters)):
            try:
                val=(self.lib_phot[list_filters[i]].get_flux(np.array(self.lib_stell.wavelength)*unit['AA'],np.array(stell)*unit['flam']))
                if list_filters[i] in self.AB:
                    pred[i]=-2.5*np.log10(val.value)-self.lib_phot[list_filters[i]].AB_zero_mag
                else:
                    pred[i]=-2.5*np.log10(val.value)-self.lib_phot[list_filters[i]].Vega_zero_mag
            except AttributeError:
                val=(self.OTHER[list_filters[i]].get_flux(np.array(self.lib_stell.wavelength)*unit['AA'],np.array(stell)*unit['flam']))
                if list_filters[i] in self.AB:
                    pred[i]=-2.5*np.log10(val)-self.OTHER[list_filters[i]].AB_zero_mag
                else:
                    pred[i]=-2.5*np.log10(val)-self.OTHER[list_filters[i]].Vega_zero_mag
        return pred

    def get_log_prob_simple(self,i,use_Z=False,estimate_error = None,no_D=False):
        """
        log prob of observations giving single star model
        """

        if i.shape[0]>3:
            if use_Z:
                logT,logL,d_p,Z=i
                g = self.gp
            else:
                logT,logL,d_p,g = i
                logT_low,logT_high = self.get_boundaries(g)
                logT = logT*(logT_high-logT_low)+logT_low
                Z = self.Z
        else:
            logT,logL,d_p=i
            g=self.gp
            Z = self.Z
        if self.use_parallax:
            d=1/d_p
        else:
            d=d_p
        pred=np.zeros(len(self.ampl))
        stell=self.lib_stell.generate_stellar_spectrum(logT,g,logL,Z)/(4*math.pi*d**2*kpc**2)
        if self.EBV!=None:
            stell=np.power(10,-0.4*self.ext)*stell

        for i in range(len(self.ampl)):
            val=(self.fil_obj[i].get_flux(self.lib_stell.wavelength.magnitude*unit['AA'],np.array(stell)*unit['flam']))
            try:
                pred[i]=val.value/self.zerop[i]
            except AttributeError:
                pred[i]=val/self.zerop[i]

        if estimate_error is None:
            errors = self.err_flux
        else:
            errors = np.zeros_like(self.err_flux)
            for i in range(len(self.err)):
                if self.err_estimate_flag[i]:
                    errors[i] = np.sqrt((pred[i]*self.error_estimate[1]*np.log(10)*0.4)**2+self.err_flux[i]**2)
                else:
                    errors[i] = self.err_flux[i]

        return_val= -np.sum((pred-self.fluxes_)**2/(2*errors**2))
        if no_D:
            return return_val
        if self.use_parallax:
            return return_val-(d_p-self.plx)**2/(2*self.e_plx**2)
        else:
            return return_val-(d-self.d)**2/(2*self.d_err**2)

    def get_log_prob_double(self,i,limit=False,estimate_error=None,no_D=False):
        """
        log prob of observations given 2 star model
        """
        logT1,logL1,logT2,logL2,d_p=i
        if limit and logT2>logT1:
            return -np.inf
        if self.use_parallax:
            d=1/d_p
        else:
            d=d_p
        pred=np.zeros(len(self.ampl))
        f1 = self.lib_stell.generate_stellar_spectrum(logT1,self.g1,logL1,self.Z1)/(4*math.pi*d**2*kpc**2)
        f2 = self.lib_stell.generate_stellar_spectrum(logT2,self.g2,logL2,self.Z2)/(4*math.pi*d**2*kpc**2)
        if self.EBV!=None:
            f1=np.power(10,-0.4*self.ext)*f1
            f2=np.power(10,-0.4*self.ext)*f2
        stell = np.stack((f1,f2))
        val_arr = np.zeros([len(self.ampl),2])
        for i in range(len(self.ampl)):
            val=(self.fil_obj[i].get_flux(self.lib_stell.wavelength.magnitude*unit['AA'],stell*unit['flam']))/self.zerop[i]
            try:
                pred[i]=(val[0].value+val[1].value)
            except AttributeError:
                pred[i]=(val[0]+val[1])
            val_arr[i] = val
        if estimate_error is None:
            errors = self.err_flux
        elif estimate_error == "1":
            flux=(self.error_estimate[0].get_flux(self.lib_stell.wavelength.magnitude*unit['AA'],stell*unit['flam']))/self.zerop[i]
            try:
                correction=(flux[0].value+flux[1].value)/flux[0].value
            except AttributeError:
                correction=(flux[0]+flux[1])/flux[0]
            
            errors = np.zeros_like(self.err_flux)
            for i in range(len(self.err)):
                if self.err_estimate_flag[i]:
                    errors[i] = np.sqrt((val_arr[i,0]*correction*np.log(10)*0.4*self.error_estimate[1])**2+self.err_flux[i]**2)
                else:
                    errors[i] = self.err_flux[i]
        
        elif estimate_error == "2":
            flux=(self.error_estimate[0].get_flux(self.lib_stell.wavelength.magnitude*unit['AA'],stell*unit['flam']))/self.zerop[i]
            try:
                correction=(flux[0].value+flux[1].value)/flux[1].value
            except AttributeError:
                correction=(flux[0]+flux[1])/flux[1]
            errors = np.zeros_like(self.err_flux)
            for i in range(len(self.err)):
                if self.err_estimate_flag[i]:
                    errors[i] = np.sqrt((val_arr[i,1]*correction*np.log(10)*0.4*self.error_estimate[1])**2+self.err_flux[i]**2)
                else:
                    errors[i] = self.err_flux[i]
        #print(errors/self.err_flux)
        return_val= -np.sum((pred-self.fluxes_)**2/(2*self.err_flux**2))
        if no_D:
            return return_val
        if self.use_parallax:
            return return_val-(d_p-self.plx)**2/(2*self.e_plx**2)
        else:
            return return_val-(d-self.d)**2/(2*self.d_err**2)


    def get_bic_double(self,**kwargs):
        chi2 = -2*self.get_log_prob_double(self.par_double,no_D=True,**kwargs)
        bic=chi2+5*np.log(len(self.ampl))
        print("chi2: ",chi2)
        print("BIC: ",bic)
        self.bicd=bic
        self.chi2d=chi2
    def get_bic_simple(self,estimate_error = None):

        if self.Z is None:
            chi2 = -self.get_log_prob_simple(self.par_single,True,estimate_error = estimate_error,no_D=True)*2
        else:
            chi2 = -self.get_log_prob_simple(self.par_single,False,estimate_error = estimate_error,no_D=True)*2
        bic=chi2+3*np.log(len(self.ampl))
        print("chi2: ",chi2)
        print("BIC: ",bic)
        self.bic_simple=bic
        self.chis = chi2

    def run_chain_double(self,num_step:int,num_burn:int,n:int,
                         progress:Optional[bool]=True,use_simple_res:Optional[bool]=False,loglrange=(-3,5),
                         rerun:Optional[bool]=False,
                         start=None,**kwargs):
        """
        run chain for double star model
        num_step - number of steps
        num_burn - number of burn-in steps
        n - number of walkers 
        progress - progres bar
        use_simple_res - whether to use values for single star model as starting point
        """
        logl_low,logl_high=loglrange
        logT1_low,logT1_high=self.get_boundaries(self.g1)
        logT2_low,logT2_high=self.get_boundaries(self.g2)
        if start is None:
            start=np.zeros([n,5])
            start[:,2]=self.to_temp(np.random.rand(n),self.g2)
            start[:,3]=np.random.rand(n)*(logl_high-logl_low)+logl_low
            if not use_simple_res:
                start[:,0]=self.to_temp(np.random.rand(n),self.g1)
                start[:,1]=np.random.rand(n)*(logl_high-logl_low)+logl_low
            else:
                start[:,0]=self.par_single[0]
                start[:,1]=self.par_single[1]
                start[:,0]+=np.random.randn(n)*0.01
                start[:,1]+=np.random.randn(n)*0.01
            if self.use_parallax:
                start[:,4]=np.random.randn(n)*self.e_plx+self.plx
            else:
                start[:,4]=np.random.randn(n)*self.d_err+self.d
            if kwargs["limit"]==True:
                if logT1_high<logT2_low:
                    raise ValueError("Wrong!")
                elif logT2_high<logT1_low:
                    pass
                else:
                    h=min(logT1_high,logT2_high)
                    l=max(logT2_low,logT1_low)
                    start[:,2]=np.random.rand(n)*(h-l)+l
                    start[:,0]=np.random.rand(n)*(logT1_high-start[:,2])+start[:,2]
        else:
            if len(start.shape)==1:
                start=np.repeat(np.array([start]),n,axis=0)+np.random.randn(n,5)*0.01

        print("starting conditions: ",start)

        bijector_list_double=[Sigmoid(logT1_low,logT1_high),Sigmoid(logl_low,logl_high),Sigmoid(logT2_low,logT2_high),Sigmoid(logl_low,logl_high),Exp()]

        sampler = emcee.EnsembleSampler(
            n, 5, (biject(bijector_list_double))(self.get_log_prob_double),kwargs=kwargs
            )
        starting=transform(start,bijector_list_double)
        sampler.run_mcmc(starting, num_step+num_burn, progress=progress)
        if rerun:
            temp=untransform(sampler.get_chain(flat=True,discard=num_burn),bijector_list_double)
            temp_tran=np.unique(temp,axis=0)
            print("sampling new starting conditions")
            id = np.random.permutation(len(temp_tran))[:n]
            new_start = temp_tran[id] + np.random.randn(n,5) * 0.001
            print("new starting conditions:")
            print(new_start)
            sampler.reset()
            sampler.run_mcmc(transform(new_start,bijector_list_double), num_step, progress=progress)
        temp=untransform(sampler.get_chain(flat=True,discard=num_burn),bijector_list_double)
        logT1_samples,logL1_samples,logT2_samples,logL2_samples,d_samples=temp.T
        print("acceptence ratio:", np.mean(sampler.acceptance_fraction))
        self.par_double=[np.median(logT1_samples),np.median(logL1_samples),np.median(logT2_samples),np.median(logL2_samples),np.median(d_samples)]
        self.par_double_container=temp.T
        ###
        self.get_bic_double(**kwargs)
        self.log_prob_chainp=sampler.get_log_prob(flat=True,discard=num_burn)


    
    def run_chain_simple(self,num_step,num_burn,n,progress=True,LogL_range=(-3,5),start=None,rerun=False,**kwargs):
        logl_low,logl_high=LogL_range
        logT_low,logT_high=self.get_boundaries(self.gp)
        bijector_list_sig=[Sigmoid(logT_low,logT_high),Sigmoid(logl_low,logl_high),Exp()]
        sampler = emcee.EnsembleSampler(
            n, 3, (biject(bijector_list_sig))(self.get_log_prob_simple),kwargs=kwargs
            )

        if start is None:
            start=np.zeros([n,3])
            start[:,0]=np.random.rand(n)*(logT_high-logT_low)+logT_low
            start[:,1]=np.random.rand(1,n)*(logl_high-logl_low)+logl_low
            if self.use_parallax:
                start[:,2]=np.random.randn(n)*self.e_plx+self.plx
            else:
                start[:,2]=np.random.randn(n)*self.d_err+self.d
        else:
            if len(start.shape)==1:
                start=np.repeat(np.array([start]),n,axis=0)+np.random.randn(n,3)*0.01
        print("starting conditions:", start)
        start_tf=transform(start,bijector_list_sig)
        sampler.run_mcmc(start_tf, num_step+num_burn, progress=progress)
        if rerun:
            temp=untransform(sampler.get_chain(flat=True,discard=num_burn),bijector_list_sig)
            temp_tran=np.unique(temp,axis=0)
            print("sampling new starting conditions")
            id = np.random.permutation(len(temp_tran))[:n]
            new_start = temp_tran[id] + np.random.randn(n,3) * 0.001
            print("new starting conditions:")
            print(new_start)
            sampler.reset()
            sampler.run_mcmc(transform(new_start,bijector_list_sig), num_step, progress=progress)
        print("acceptance ratio",np.mean(sampler.acceptance_fraction))
        states=untransform(sampler.get_chain(flat=True,discard=num_burn),bijector_list_sig)
        logT_samples,logL_samples,val_samples=states.T
        self.par_single=np.array([np.median(logT_samples),np.median(logL_samples),np.median(val_samples)])
        self.par_single_container=states.T
        self.get_bic_simple(**kwargs)
        print("parameters:",self.par_single)
        self.log_prob_chain=sampler.get_log_prob(flat=True,discard=num_burn)

    def run_chain_simple_with_g(self,num_step,num_burn,n,g_range,progress=True,LogL_range=(-3,5),start=None,rerun=False):
        logl_low,logl_high=LogL_range
        g_low,g_high=g_range
        self.gp=None
        bijector_list_sig=[Sigmoid(0,1),Sigmoid(logl_low,logl_high),Exp(),Sigmoid(g_low,g_high)]
        sampler = emcee.EnsembleSampler(
            n, 4, (biject(bijector_list_sig))(self.get_log_prob_simple)
            )
        if start is None:
            start=np.zeros([n,4])
            start[:,0]=np.random.rand(n)
            start[:,1]=np.random.rand(1,n)*(logl_high-logl_low)+logl_low
            if self.use_parallax:
                start[:,2]=np.random.randn(n)*self.e_plx+self.plx
            else:
                start[:,2]=np.random.randn(n)*self.d_err+self.d
            start[:,3]=np.random.rand(n)*(g_high-g_low)+g_low
        else:
            if len(start.shape)==1:
                logT_low,logT_high=self.get_boundaries(start[3])
                start[0]=(start[0]-logT_low)/(logT_high-logT_low)
                start=np.repeat(np.array([start]),n,axis=0)+np.random.randn(n,4)*0.01
        print("starting conditions:", start)
        start_tf=transform(start,bijector_list_sig)
        sampler.run_mcmc(start_tf, num_step+num_burn, progress=progress)
        print("acceptance ratio",np.mean(sampler.acceptance_fraction))
        if rerun:
            temp=untransform(sampler.get_chain(flat=True,discard=num_burn),bijector_list_sig)
            temp_tran=np.unique(temp,axis=0)
            print("sampling new starting conditions")
            id = np.random.permutation(len(temp_tran))[:n]
            new_start = temp_tran[id] + np.random.randn(n,4) * 0.01
            print("new starting conditions:")
            print(new_start)
            sampler.reset()
            sampler.run_mcmc(transform(new_start,bijector_list_sig), num_step, progress=progress)
        states=untransform(sampler.get_chain(flat=True,discard=num_burn),bijector_list_sig)
        for i in range(states.shape[0]):
            logT_low,logT_high=self.get_boundaries(states[i][3])
            states[i][0]=states[i][0]*(logT_high-logT_low)+logT_low
        self.par_single=np.median(states,0)
        self.par_single_container=states.T
        self.get_bic_simple()
        print("parameters:",self.par_single)
        self.log_prob_chain=sampler.get_log_prob(flat=True,discard=num_burn)


    def run_chain_simple_with_Z(self,num_step,num_burn,n,Z_range,progress=True,LogL_range=(-3,5),start=None,rerun=False,estimate_error = None):
        logl_low,logl_high=LogL_range
        z_low,z_high=Z_range
        logT_low,logT_high=self.get_boundaries(self.gp)
        Z_start=self.Z
        self.Z=None
        bijector_list_sig=[Sigmoid(logT_low,logT_high),Sigmoid(logl_low,logl_high),Exp(),Sigmoid(z_low,z_high)]
        sampler = emcee.EnsembleSampler(
            n, 4, (biject(bijector_list_sig))(self.get_log_prob_simple),
            kwargs={"use_Z":True,"estimate_error":estimate_error}
            )
        if start is None:
            start=np.zeros([n,4])
            start[:,0]=np.random.rand(n)*(logT_high-logT_low)+logT_low
            start[:,1]=np.random.rand(1,n)*(logl_high-logl_low)+logl_low
            if self.use_parallax:
                start[:,2]=np.random.randn(n)*self.e_plx+self.plx
            else:
                start[:,2]=np.random.randn(n)*self.d_err+self.d
            start[:,3]=Z_start*(1+np.random.randn(n)/20)     #np.random.rand(n)*(z_high-z_low)+z_low
        else:
            if len(start.shape)==1:
                logT_low,logT_high=self.get_boundaries(start[3])
                start=np.repeat(np.array([start]),n,axis=0)+np.random.randn(n,4)*0.01
        print("starting conditions:", start)
        start_tf=transform(start,bijector_list_sig)
        sampler.run_mcmc(start_tf, num_step+num_burn, progress=progress)
        print("acceptance ratio",np.mean(sampler.acceptance_fraction))
        if rerun:
            temp=untransform(sampler.get_chain(flat=True,discard=num_burn),bijector_list_sig)
            temp_tran=np.unique(temp,axis=0)
            print("sampling new starting conditions")
            id = np.random.permutation(len(temp_tran))[:n]
            new_start = temp_tran[id] + np.random.randn(n,4) * 0.001
            print("new starting conditions:")
            print(new_start)
            sampler.reset()
            sampler.run_mcmc(transform(new_start,bijector_list_sig), num_step, progress=progress)
        states=untransform(sampler.get_chain(flat=True,discard=num_burn),bijector_list_sig)
        self.par_single=np.median(states,0)
        self.par_single_container=states.T
        self.get_bic_simple()
        print("parameters:",self.par_single)
        self.log_prob_chain=sampler.get_log_prob(flat=True,discard=num_burn)


   
    def plot_measurments(self,ax,plot_fwhm=False,plot_ebv=True,limit=2,errors=None):
        N = len(self.ampl)

        lam, lamF_lam, logerr = [], [], []
        lamFlam,lamFlam_err = [], []
        fwhm=[]
        for i in range(N):
            if self.filters[i] in self.AB:
                AB_if=True
            else:
                AB_if=False
            if errors is None:
                _lam, _F_nu, _F_lam, _lamF_lam, _frac_error = mag_to_flux_pyphot(self.fil_obj[i], self.ampl[i], self.err[i],AB_if)
            else:
                #print(errors)
                _lam, _F_nu, _F_lam, _lamF_lam, _frac_error = mag_to_flux_pyphot(self.fil_obj[i], self.ampl[i], errors[i],AB_if)
            lam.append(_lam)
            lamF_lam.append(3+np.log10(_lamF_lam))
            logerr.append(_frac_error/np.log(10.0))
            lamFlam.append(_lamF_lam*1000)
            lamFlam_err.append(_lamF_lam*1000*_frac_error)
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
        lamF_lam=np.array(lamF_lam)
        ax.set_ylim(min(lamF_lam)-0.5,max(lamF_lam)+limit)
        ax.set_xlim(0.1,10)
        if self.EBV is not None and plot_ebv:
            plt.figtext(0.65,0.05,r"$E(B-V)={0:.3f}$".format(self.EBV))
        if len(lam)>0:
            if plot_fwhm:
                ax.errorbar(x=lam, y=lamF_lam, yerr=logerr,xerr=fwhm, fmt='o', mfc='navy', color='navy', ms=3,  capsize=2,label="measurements")
            else:
                ax.errorbar(x=lam, y=lamF_lam, yerr=logerr, fmt='o', mfc='navy', color='navy', ms=3,capsize=2,label="measurements")
    
    def plot_dist_simple(self,FWHM=False,ax=None,plot_bic=True,plot_chi2=False,save=True,plot_label=True,plot_d=True,estimate_error=False,**kwargs):
        if ax is None:
            fig = plt.figure(figsize=(6,4))
            ax = plt.axes()
        if self.gp is None:
            g=self.par_single[3]
        else:
            g=self.gp
        if self.Z is None:
            Z=self.par_single[3]
        else:
            Z=self.Z
        if self.use_parallax:
            flux=self.lib_stell.generate_stellar_spectrum(self.par_single[0],g,self.par_single[1],Z)*self.par_single[2]**2/(4*math.pi*kpc**2)
        else:
            flux=self.lib_stell.generate_stellar_spectrum(self.par_single[0],g,self.par_single[1],Z)/(4*math.pi*self.par_single[2]**2*kpc**2)
        if self.EBV!=None:
            flux=np.power(10,-0.4*self.ext)*flux

        pred = np.zeros_like(self.ampl)
        for i in range(len(self.ampl)):
            val=(self.fil_obj[i].get_flux(self.lib_stell.wavelength.magnitude*unit['AA'],np.array(flux)*unit['flam']))
            try:
                pred[i]=val.value/self.zerop[i]
            except AttributeError:
                pred[i]=val/self.zerop[i]

        if estimate_error is None:
            errors = self.err_flux
        else:
            errors = np.zeros_like(self.err_flux)
            for i in range(len(self.err)):
                if self.err_estimate_flag[i]:
                    errors[i] = np.sqrt(self.error_estimate[1]**2+self.err[i]**2)
                else:
                    errors[i] = self.err[i]
        self.plot_measurments(ax,FWHM,errors=errors,**kwargs)

        r_sample=self.lib_stell.get_radius(self.par_single[1],self.par_single_container[0])
        if self.use_parallax:
            d_samples=1/self.par_single_container[2]
        else:
            d_samples=self.par_single_container[2]
        param=[10**self.par_single[0],10**np.quantile(self.par_single_container[0],0.84)-10**self.par_single[0],10**self.par_single[0]-10**np.quantile(self.par_single_container[0],0.16),
        np.median(d_samples),np.median(r_sample)/Rs,np.quantile(r_sample/Rs,0.84)-np.median(r_sample/Rs),np.median(r_sample)/Rs-np.quantile(r_sample/Rs,0.16),self.Z
        ]
        if self.Z is not None:
            label=r"$T={0:.0f}^{{+{1:.0f}}}_{{-{2:.0f}}}$ K, $R={4:.3f}^{{+{5:.2g}}}_{{-{6:.2g}}}$ $R_{{\odot}}$, $Z={7:.3f}$".format(*param)
        else:
            new_param=[*param[:-1],self.par_single[3],
                       np.quantile(self.par_single_container[3],0.84)-np.median(self.par_single_container[3]),self.par_single[3]-np.quantile(self.par_single_container[3],0.16)]
            label=r"$T={0:.0f}^{{+{1:.0f}}}_{{-{2:.0f}}}$ K, $R={4:.3f}^{{+{5:.2g}}}_{{-{6:.2g}}}$ $R_{{\odot}}$, $Z={7:.4f}^{{+{8:.2g}}}_{{-{9:.2g}}}$".format(*new_param)
        ax.plot(np.array(self.lib_stell.wavelength)/10**4,np.log10(np.array(flux)*np.array(self.lib_stell.wavelength)),
        label=label,color="orange")
        ax.set_xscale('log')
        ax.legend()
        if plot_label:
            ax.set_xlabel(r'$\lambda$ [$\mu\textrm{m}$]')
            ax.set_ylabel(r'log $\lambda F_{\lambda}$ (erg cm$^{-2}$s$^{-1}$)')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_title(self.name)
        if plot_bic:
            plt.figtext(0.2,0.05,"BIC: {0:.1f}".format(self.bic_simple))
        if plot_chi2:
            plt.figtext(0.2,0.05,"$\chi^2$= {0:.1f} N = {1:.0f}".format(self.chis,len(self.filters)))
        if self.gp is None:
            plt.figtext(0.2,0.1,"$\log(g)={0:.2f}^{{+{2:.2f}}}_{{-{1:.2f}}}$".format(np.median(self.par_single_container[3]),
                                                                                np.median(self.par_single_container[3])-np.quantile(self.par_single_container[3],0.16),
                                                                                np.quantile(self.par_single_container[3],0.84)-np.median(self.par_single_container[3])))
        if self.EBV is None:
            high=0.05
        else:
            high=0.1
        if plot_d:
            plt.figtext(0.65,high,"$d={0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$ kpc".format(np.median(d_samples),
                np.quantile(d_samples,0.84)-np.median(d_samples),
                np.median(d_samples)-np.quantile(d_samples,0.16)
                ))
        if save==True:
            plt.tight_layout()
            plt.savefig(self.name+"_simple_emcee.png",dpi=500)

    def plot_corner_simple(self,n_l=2.2,bi=20,):
        data=self.par_single_container.T
        upper=list(map(lambda x:(n_l+2)/2*np.quantile(x,0.84)-np.quantile(x,0.16)*n_l/2,self.par_single_container))
        lower=list(map(lambda x:(n_l+2)/2*np.quantile(x,0.16)-np.quantile(x,0.84)*n_l/2,self.par_single_container))
        index=np.all(np.logical_and(data<upper,data>lower).T,axis=0)
        data=data[index,:]
        bins=list(map(lambda x:int(bi/(upper[x]-lower[x])*(max(data[:,x])-min(data[:,x]))),range(len(upper))))

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
        if len(bins)>3:
            if self.gp is None:
                labels.append("log g")
            else:
                labels.append("$Z$")
        figure = corner.corner(
            data,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            title_fmt=".3f",
            bins=bins
        )
        
        axes = np.array(figure.axes).reshape((len(upper), len(upper)))
        for i in range(len(upper)):
            for j in range(i+1):
                ax=axes[i,j]
                ax.set_xlim(lower[j],upper[j])
                if j<i:
                    ax.set_ylim(lower[i],upper[i])
        
        figure.suptitle(self.name)
        plt.savefig(self.name+"_simple_corner_emcee.png",dpi=500)


    def plot_dist_double(self,FWHM=False,ax=None,plot_bic=True,estimate_error=None):
        if ax is None:
            fig = plt.figure(figsize=(6,4))
            ax=plt.axes()
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
        flux1=self.lib_stell.generate_stellar_spectrum(self.par_double[0],self.g1,self.par_double[1],self.Z1)/(4*math.pi*np.median(d_samples)**2*kpc**2)
        flux2=self.lib_stell.generate_stellar_spectrum(self.par_double[2],self.g2,self.par_double[3],self.Z2)/(4*math.pi*np.median(d_samples)**2*kpc**2)
        if self.EBV!=None:
            flux1=np.power(10,-0.4*self.ext)*flux1
            flux2=np.power(10,-0.4*self.ext)*flux2
        stell = np.stack((flux1,flux2))
        val_arr = np.zeros([len(self.ampl),2])
        pred = np.zeros_like(self.ampl)
        for i in range(len(self.ampl)):
            val=(self.fil_obj[i].get_flux(self.lib_stell.wavelength.magnitude*unit['AA'],stell*unit['flam']))/self.zerop[i]
            try:
                pred[i]=(val[0].value+val[1].value)
            except AttributeError:
                pred[i]=(val[0]+val[1])
            val_arr[i] = val
        if estimate_error is None:
            errors = self.err_flux
        elif estimate_error == "1":
            flux=(self.error_estimate[0].get_flux(self.lib_stell.wavelength.magnitude*unit['AA'],stell*unit['flam']))/self.zerop[i]
            try:
                correction=(flux[0].value+flux[1].value)/flux[0].value
            except AttributeError:
                correction=(flux[0]+flux[1])/flux[0]
            
            errors = np.zeros_like(self.err_flux)
            for i in range(len(self.err)):
                if self.err_estimate_flag[i]:
                    errors[i] = np.sqrt((val_arr[i,0]*correction/pred[i]*self.error_estimate[1])**2+self.err[i]**2)
                else:
                    errors[i] = self.err[i]
        
        elif estimate_error == "2":
            flux=(self.error_estimate[0].get_flux(self.lib_stell.wavelength.magnitude*unit['AA'],stell*unit['flam']))/self.zerop[i]
            try:
                correction=(flux[0].value+flux[1].value)/flux[1].value
            except AttributeError:
                correction=(flux[0]+flux[1])/flux[1]
            errors = np.zeros_like(self.err_flux)
            for i in range(len(self.err)):
                if self.err_estimate_flag[i]:
                    errors[i] = np.sqrt((val_arr[i,1]*correction/pred[i]*self.error_estimate[1])**2+self.err[i]**2)
                else:
                    errors[i] = self.err[i]

        self.plot_measurments(ax,FWHM,errors=errors)
        #for i in range(len(self.filters)):
        #    print(self.filters[i],errors[i]/self.err[i])
        ax.plot(np.array(self.lib_stell.wavelength)/10**4,np.log10(np.array(flux2)*np.array(self.lib_stell.wavelength)),
        label=r"$T={0:.0f}^{{+{1:.0f}}}_{{-{2:.0f}}}$ K, $R={3:.2f}^{{+{4:.2g}}}_{{-{5:.2g}}}$ $R_{{\odot}}$".format(*param2),color="red")
        ax.plot(np.array(self.lib_stell.wavelength)/10**4,np.log10(np.array(flux1)*np.array(self.lib_stell.wavelength)),
        label=r"$T={0:.0f}^{{+{1:.0f}}}_{{-{2:.0f}}}$ K, $R={3:.2f}^{{+{4:.2g}}}_{{-{5:.2g}}}$ $R_{{\odot}}$".format(*param1),color="yellow")
        ax.plot(np.array(self.lib_stell.wavelength)/10**4,np.log10(np.array(flux1+flux2)*np.array(self.lib_stell.wavelength)),label="combined",color="grey")
        
        ax.set_xscale('log')
        plt.legend()
        flux=flux1+flux2
        ax.set_xlabel(r'$\lambda$ [$\mu\textrm{m}$]')
        ax.set_ylabel(r'log $\lambda F_{\lambda}$ (erg cm$^{-2}$s$^{-1}$)')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.title(self.name)
        if plot_bic:
            plt.figtext(0.2,0.05,"BIC: {0:.1f}".format(self.bicd))
        else:
            plt.figtext(0.2,0.05,"$\chi^2$= {0:.1f} N = {1:.0f}".format(self.chi2d,len(self.filters)))
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

    def plot_corner_double(self,n_l=2.2,bi=20,):
        data=self.par_double_container.T
        upper=[(n_l+2)/2*np.quantile(self.par_double_container[0],0.84)-np.quantile(self.par_double_container[0],0.16)*n_l/2,(n_l+2)/2*np.quantile(self.par_double_container[1],0.84)-np.quantile(self.par_double_container[1],0.16)*n_l/2,(n_l+2)/2*np.quantile(self.par_double_container[2],0.84)-np.quantile(self.par_double_container[2],0.16)*n_l/2,
        (n_l+2)/2*np.quantile(self.par_double_container[3],0.84)-np.quantile(self.par_double_container[3],0.16)*n_l/2,(n_l+2)/2*np.quantile(self.par_double_container[4],0.84)-np.quantile(self.par_double_container[4],0.16)*n_l/2]
        lower=[(n_l+2)/2*np.quantile(self.par_double_container[0],0.16)-np.quantile(self.par_double_container[0],0.84)*n_l/2,(n_l+2)/2*np.quantile(self.par_double_container[1],0.16)-np.quantile(self.par_double_container[1],0.84)*n_l/2,(n_l+2)/2*np.quantile(self.par_double_container[2],0.16)-np.quantile(self.par_double_container[2],0.84)*n_l/2,
        (n_l+2)/2*np.quantile(self.par_double_container[3],0.16)-np.quantile(self.par_double_container[3],0.84)*n_l/2,(n_l+2)/2*np.quantile(self.par_double_container[4],0.16)-np.quantile(self.par_double_container[4],0.84)*n_l/2]
        t_min2,t_max2=self.get_boundaries(self.g2)
        t_min1,t_max1=self.get_boundaries(self.g1)
        upper[0]=min(upper[0],t_max1)
        #upper[1]=min(upper[1],5)
        upper[2]=min(upper[2],t_max2)
        #upper[3]=min(upper[3],5)
        lower[0]=max(lower[0],t_min1)
        #lower[1]=max(lower[1],-3)
        lower[2]=max(lower[2],t_min2)
        #lower[3]=max(lower[3],-3)
        index=np.all(np.logical_and(data<upper,data>lower).T,axis=0)
        data=data[index,:]
        bins=list(map(lambda x:int(bi/(upper[x]-lower[x])*(max(data[:,x])-min(data[:,x]))),range(len(upper))))

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
        print(lower,upper)
        figure = corner.corner(
            data,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            title_fmt=".3f",
            bins=bins
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

    def plot_dist_simple_density(self,num_samples,FWHM=False):
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
        plt.savefig(self.name.replace(" ","_")+"_simple_emcee.png",dpi=500)
        print("parameters:",self.par_single)


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
        plt.savefig(self.name.replace(" ","_")+"_double_emcee.png",dpi=500)
        print("parameters:",self.par_double)
    

    def save(self,name=None):
        """
        save parameters of model to h5 file
        """
        if name is None:
            name=self.name.replace(" ","_")+".h5"
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
            file.create_dataset("plx",data=np.array([self.plx,self.e_plx]))
        if self.gp is not None:
            file.create_dataset("gp",data=(self.gp,))
        file.create_dataset("g1",data=(self.g1,))
        file.create_dataset("g2",data=(self.g2,))
        if self.Z is not None:
            file.create_dataset("Z",data=(self.Z,))
        if self.EBV is not None:
            file.create_dataset("EBV",data=(self.EBV,))
        file.close()

    def load(self,name=None):
        """
        load chain parameters
        """
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
        if "gp" in file.keys():
            self.gp=np.array(file.get("gp"))[0]
        self.g1=np.array(file.get("g1"))[0]
        self.g2=np.array(file.get("g2"))[0]
        if "Z" in file.keys():
            self.Z=np.array(file.get("Z"))[0]
        if "EBV" in file.keys():
            self.set_EBV(np.array(file.get("EBV"))[0])
        file.close()
