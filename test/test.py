from Iris import Star,Galactic
from mwdust import Combined19
from astropy import units as u
from astropy.coordinates import SkyCoord
name="Giraffe"
ra=63.131407404
dec=67.646834346
t=Star(name,ra,dec,catalog=Galactic)
t.get_parallax()
t.get_all()
t.prepare_data()
for cat,fil,err in zip(t.filters,t.ampl,t.err):
    print(cat,fil,err)
ext_model=Combined19()
galactic_coords=SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs').galactic
extinction=ext_model(galactic_coords.l.value,galactic_coords.b.value,1/t.plx)[0]
t.set_EBV(extinction)
t.run_chain_simple(3000,500,8)
t.plot_dist_simple()
t.rerun_chain_double(3000,500,16)
t.plot_dist_double()