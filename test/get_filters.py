"""
This script shows how to use Iris to reconstruct SED fit of Gaia BH1 object from https://arxiv.org/abs/2209.06833
"""

from Iris import Star,Galactic
import numpy as np
name="Gaia BH1"
ra=262.171207276
dec=-0.581091963
orginal_data=Galactic.copy()
del orginal_data["I/345/gaia2"] # we do not want overcrowded optical photometry so entries like Gaia, VHS and SkyMapper are disabled
del orginal_data["II/367/vhs_dr5"]
del orginal_data["II/358/smss"]
t=Star(name,ra=ra,dec=dec,catalog=orginal_data)
t.dis_norm=2
t.get_parallax()
t.get_all(False)
import json
data_dic={}
for name,val in zip(t.filters,t.ampl):
    data_dic[name]=val
data_dic["EBV"]=0.3
with open("GAIA_BH1.json","w") as f:
    f.write(json.dumps(data_dic))