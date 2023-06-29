"""
default calatlog used to find data
format- name of Vizier catalog is mapped to list containing 
1.list of columns in format magnitude, magnitude_err
2.list of name of filters to save magnitudes
3.radius of cone search measured in arcsec
"""

MagCloud={
    "I/345/gaia2":[["Gmag","e_Gmag","BPmag","e_BPmag","RPmag","e_RPmag"],
    ["G_Gaia","BP_Gaia","RP_Gaia"],0.4],
    "II/246/out":[["Hmag","e_Hmag","Kmag","e_Kmag","Jmag","e_Jmag"],
    ["H_2MASS","K_2MASS","J_2MASS"],0.4],
    "II/281/2mass6x":[["Hmag","e_Hmag","Kmag","e_Kmag","Jmag","e_Jmag"],
    ["H_2MASS","K_2MASS","J_2MASS"],0.4],
    "II/351/vmc_dr4":[["KSAPERMAG3","KSAPERMAG3ERR","JAPERMAG3","JAPERMAG3ERR","YAPERMAG3","YAPERMAG3ERR"],
    ["K_VISTA","J_VISTA","Y_VISTA"],0.4],
    "II/367/vhs_dr5":[["KSAPERMAG3","KSAPERMAG3ERR","JAPERMAG3","JAPERMAG3ERR","YAPERMAG3","YAPERMAG3ERR"],
    ["K_VISTA","J_VISTA","Y_VISTA"],0.4],
    "II/359/vhs_dr4":[["KSAPERMAG3","KSAPERMAG3ERR","JAPERMAG3","JAPERMAG3ERR","YAPERMAG3","YAPERMAG3ERR"],
    ["K_VISTA","J_VISTA","Y_VISTA"],0.4],
    "II/305/catalog":[["[3.6]","e_[3.6]","[4.5]","e_[4.5]","[5.8]","e_[5.8]","[8.0]","e_[8.0]"],
    ["3.6","4.5","5.8","8.0"],0.5],
    "II/305/archive":[["[3.6]","e_[3.6]","[4.5]","e_[4.5]","[5.8]","e_[5.8]","[8.0]","e_[8.0]"],
    ["3.6","4.5","5.8","8.0"],0.5],
     "II/328/allwise":[["W1mag","e_W1mag","W2mag","e_W2mag","W3mag","e_W3mag","W4mag","e_W4mag"],
    ["W1","W2","W3","W4"],0.3],
    "II/311/wise":[["W1mag","e_W1mag","W2mag","e_W2mag","W3mag","e_W3mag","W4mag","e_W4mag"],
    ["W1","W2","W3","W4"],0.3],
    "II/365/catwise":[["w1mpro_pm","w1sigmpro_pm","w2mpro_pm","w2sigmpro_pm","w3mpro_pm","w3sigmpro_pm"],["W1","W2","W3"],0.5],
    "II/358/smss":[["u_psf","e_u_psf","v_psf","e_v_psf","r_psf","e_r_psf","i_psf","e_i_psf","g_psf","e_g_psf","z_psf","e_z_psf"],
    ["u_SM","v_SM","r_SM","i_SM","g_SM","z_SM"],0.3],
    #,
    #"J/AcA/48/147/smc_sc":[["Vmag","e_Vmag","Bmag","e_Bmag","Imag","e_Imag"],
    #["V","B","I"]]
    "J/AJ/123/855/table1":[["Vmag","e_Vmag","Bmag","e_Bmag","Imag","e_Imag","Umag","e_Umag"],
    ["V","B","I","U"],0.3],
    "J/AJ/128/1606/lmcps":[["Vmag","e_Vmag","Bmag","e_Bmag","Imag","e_Imag","Umag","e_Umag"],
    ["V","B","I","U"],0.3],
    "II/228A/denisLMC":
    [["Imag","e_Imag","Jmag","e_Jmag","Kmag","e_Kmag"],
    ["DENIS_I","DENIS_J","DENIS_Ks"],0.3],
    "II/228A/denisSMC":
    [["Imag","e_Imag","Jmag","e_Jmag","Kmag","e_Kmag"],
    ["DENIS_I","DENIS_J","DENIS_Ks"],0.3],
    "B/denis/denis":[["Imag","e_Imag","Jmag","e_Jmag","Kmag","e_Kmag"],
    ["DENIS_I","DENIS_J","DENIS_Ks"],0.3],
    "II/335/galex_ais":[["FUVmag","e_FUVmag","NUVmag","e_NUVmag"],["FUV","NUV"],2.0],
    "II/312/ais":[["FUV","e_FUV","NUV","e_NUV"],["FUV","NUV"],2.0],
    "II/312/mis":[["FUV","e_FUV","NUV","e_NUV"],["FUV","NUV"],2.0],
    "II/356/xmmom41s":[["UVW2mAB ","e_UVW2mAB ","UVM2mAB","e_UVM2mAB","UVW1mAB","e_UVW1mAB "],["UVW2","UVM2","UVW1"],0.6],
    "II/370/xmmom5s":[["UVW2mAB ","e_UVW2mAB ","UVM2mAB","e_UVM2mAB","UVW1mAB","e_UVW1mAB "],["UVW2","UVM2","UVW1"],0.6],
    "II/340/xmmom2_1":[["UVW2mAB ","e_UVW2mAB ","UVM2mAB","e_UVM2mAB","UVW1mAB","e_UVW1mAB "],["UVW2","UVM2","UVW1"],0.6],
    "II/288/out":[["Hmag","e_Hmag","Kmag","e_Kmag","Jmag","e_Jmag"],
    ["H_IRSF","Ks_IRSF","J_IRSF"],0.3]
}

Galactic={
    "I/345/gaia2":[["Gmag","e_Gmag","BPmag","e_BPmag","RPmag","e_RPmag"],
    ["G_Gaia","BP_Gaia","RP_Gaia"],0.3],
    "II/246/out":[["Hmag","e_Hmag","Kmag","e_Kmag","Jmag","e_Jmag"],
    ["H_2MASS","K_2MASS","J_2MASS"],0.3],
    "II/281/2mass6x":[["Hmag","e_Hmag","Kmag","e_Kmag","Jmag","e_Jmag"],
    ["H_2MASS","K_2MASS","J_2MASS"],0.3],
    "II/367/vhs_dr5":[["KSAPERMAG3","KSAPERMAG3ERR","JAPERMAG3","JAPERMAG3ERR","YAPERMAG3","YAPERMAG3ERR"],
    ["K_VISTA","J_VISTA","Y_VISTA"],0.3],
    "II/348/vvv2":[["KSAPERMAG3","KSAPERMAG3ERR","JAPERMAG3","JAPERMAG3ERR","YAPERMAG3","YAPERMAG3ERR"],
    ["K_VISTA","J_VISTA","Y_VISTA"],0.3],
    "II/293/glimpse":[["3.6mag","e_3.6mag","4.5mag","e_4.5mag","5.8mag","e_5.8mag","8.0mag","e_8.0mag"],
    ["3.6","4.5","5.8","8.0"],0.5],
    "II/328/allwise":[["W1mag","e_W1mag","W2mag","e_W2mag","W3mag","e_W3mag","W4mag","e_W4mag"],
    ["W1","W2","W3","W4"],0.3],
    "II/311/wise":[["W1mag","e_W1mag","W2mag","e_W2mag","W3mag","e_W3mag","W4mag","e_W4mag"],
    ["W1","W2","W3","W4"],0.3],
    "II/365/catwise":[["w1mpro_pm","w1sigmpro_pm","w2mpro_pm","w2sigmpro_pm","w3mpro_pm","w3sigmpro_pm"],["W1","W2","W3"],0.5],
    "II/358/smss":[["u_psf","e_u_psf","v_psf","e_v_psf","r_psf","e_r_psf","i_psf","e_i_psf","g_psf","e_g_psf","z_psf","e_z_psf"],
    ["u_SM","v_SM","r_SM","i_SM","g_SM","z_SM"],0.3],
    "J/AJ/123/855/table1":[["Vmag","e_Vmag","Bmag","e_Bmag","Imag","e_Imag","Umag","e_Umag"],
    ["V","B","I","U"],0.3],
    "B/denis/denis":[["Imag","e_Imag","Jmag","e_Jmag","Kmag","e_Kmag"],
    ["DENIS_I","DENIS_J","DENIS_Ks"],0.3],
    "II/335/galex_ais":[["FUVmag","e_FUVmag","NUVmag","e_NUVmag"],["FUV","NUV"],2.0],
    "II/312/ais":[["FUV","e_FUV","NUV","e_NUV"],["FUV","NUV"],2.0],
    "II/312/mis":[["FUV","e_FUV","NUV","e_NUV"],["FUV","NUV"],2.0],
    "II/356/xmmom41s":[["UVW2mAB ","e_UVW2mAB ","UVM2mAB","e_UVM2mAB","UVW1mAB","e_UVW1mAB "],["UVW2","UVM2","UVW1"],0.6],
    "II/370/xmmom5s":[["UVW2mAB ","e_UVW2mAB ","UVM2mAB","e_UVM2mAB","UVW1mAB","e_UVW1mAB "],["UVW2","UVM2","UVW1"],0.6],
    "II/340/xmmom2_1":[["UVW2mAB ","e_UVW2mAB ","UVM2mAB","e_UVM2mAB","UVW1mAB","e_UVW1mAB "],["UVW2","UVM2","UVW1"],0.6],
    "J/AN/336/590/varsum":[["rmag","s_rmag","imag","s_imag"],["SDSS_r","SDSS_i"],0.4],
    "II/339/uvotssc1":[["UVW2_ABMAG","UVW2_MAG_ERR","UVM2_ABMAG","UVM2_MAG_ERR","UVW1_ABMAG","UVW1_MAG_ERR"],["SWIFT_UVW2","SWIFT_UVM2","SWIFT_UVW1"],0.5],
    "II/336/apass9":[["Vmag","e_Vmag","Bmag","e_Bmag","g_mag","e_g'mag","r_mag","e_r'mag","i_mag","e_i'mag"],["V","B","SDSS_g","SDSS_r","SDSS_i"],0.4],
    "II/349/ps1":[["gMeanPSFMag","gMeanPSFMagErr","rMeanPSFMag","rMeanPSFMagErr","iMeanPSFMag","iMeanPSFMagErr","zMeanPSFMag","zMeanPSFMagErr",
                   "yMeanPSFMag","yMeanPSFMagErr"],
                  ["g_PS1","r_PS1","i_PS1","z_PS1","y_PS1"],0.5]
}