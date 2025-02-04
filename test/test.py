"""
This script shows how to use Iris to reconstruct SED fit of Gaia BH1 object from https://arxiv.org/abs/2209.06833
"""

from Iris import Star,Galactic
import numpy as np
from pystellibs import Kurucz
name = "Gaia BH1"
ra = 262.171207276
dec = -0.581091963
orginal_data = Galactic.copy()
del orginal_data["II/367/vhs_dr5"]
t=Star(name,ra = ra,dec = dec,catalog = orginal_data)
t.dis_norm = 2
t.get_parallax()
t.get_all(False)
t.delete("WISE_RSR_W4")
t.prepare_data()
t.list_filters()

#t.get_ebv_green19() currently Bayestar not working
t.set_EBV(0.343)
t.gp = 4.5
t.run_chain_simple(3000,200,32)
t.plot_dist_simple(use_AB = True) # plot magnitudes on AB scale
t.plot_corner_simple()
