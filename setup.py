from setuptools import setup, find_packages,find_namespace_packages

setup(
    name = "Iris",
    version = "1.0",
    author = "Mateusz Kapusta",
    author_email = "mr.kapusta@student.uw.edu.pl",
    packages = find_namespace_packages(where = "src"),
    package_dir = {"":"src"},
    package_data = {"Iris.filters":["*.dat","*.csv"]},
    install_requires = ['numpy','scipy','matplotlib','seaborn'
                        ,'emcee','dustmaps','astropy','astroquery',
                        'pyphot','pyvo','h5py','corner',"extinction"],
    description = "Python library for assembling SED",
    #include_package_data = True,
    license = "MIT"
)
