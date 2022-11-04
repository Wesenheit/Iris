from Iris import Star,default
name="GD1070.18.22288"
t=Star(name,257.007546,-41.048747,catalog=default)
t.get_parallax()
t.get_all()
t.prepare_data()
t.set_EBV(0.2644)
t.run_chain_simple(3000,500,8)
t.run_chain_double(3000,500,16)
t.plot_dist_simple()
t.plot_dist_double()