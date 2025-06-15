import numpy
from matplotlib import pyplot
from scipy.stats import norm

# Simulate SDE with drift function f and noise amplitude g for arbitrary time steps.
def forward_sde_simulation(x0, nsteps, dt, f, g, params):
    # Initialize time and a stochastic trajectory.
    t = 0
    x_traj = numpy.zeros((nsteps + 1, *x0.shape))
    x_traj[0] = x0

    # Perform many Euler-Maruyama time steps.
    for i in range(nsteps):
        random_normal = numpy.random.randn(*x0.shape)
        x_traj[i + 1] = x_traj[i] + f(x_traj[i], t, params) * dt + \
            g(x_traj[i], t, params) * random_normal * numpy.sqrt(dt)
        t = t + dt

    return x_traj

# Drift function for diffusion (returns zeros)
def f_diff_simple(x, t, params):
    return numpy.zeros((*x.shape,))


# Noise amplitude for diffusion (constant)
def g_diff_simple(x, t, params):
    sigma = params['sigma']
    return sigma*numpy.ones((*x.shape,))

# Exact transition probability for 1D diffusion.
def transition_probability_diffusion_exact(x, t, params):
    x0, sigma = params['x0'], params['sigma']
    # pdf of normal distribution with mean x0 and variance (sigma^2)*t
    pdf = norm.pdf(x, loc=x0, scale=numpy.sqrt((sigma**2) * t))
    return pdf

# noise amplitude for 1D diffusion
sigma = 1
num_samples = 1000
# initial condition for diffusion
x0 = numpy.zeros(num_samples)
# number of simulation steps
nsteps = 2000
# size of small time steps
dt = 0.001
T = nsteps * dt
t = numpy.linspace(0, T, nsteps + 1)
params = {'sigma': sigma, 'x0': x0, 'T': T}

x_traj = forward_sde_simulation(x0, nsteps, dt, f_diff_simple, g_diff_simple, params)

# Plot initial distribution (distribution before diffusion).
pyplot.hist(x_traj[0], bins=50, density=True)
pyplot.grid()
pyplot.show()

# Compute exact transition probability.
x_f_min, x_f_max = numpy.amin(x_traj[-1]), numpy.amax(x_traj[-1])
num_xf = 1000
x_f_arg = numpy.linspace(x_f_min, x_f_max, num_xf)
pdf_final = transition_probability_diffusion_exact(x_f_arg, T, params)

# Plot final distribution (distribution after diffusion).
pyplot.hist(x_traj[-1], bins=100, density=True, label='Simulation')
pyplot.plot(x_f_arg, pdf_final, label='Exact', color='red')
pyplot.grid()
pyplot.legend()
pyplot.show()

# Plot some trajectories.
sample_trajectories = [0, 1, 2, 3, 4]
for s in sample_trajectories:
    pyplot.plot(t, x_traj[:, s], label=f'Sample {s}')
pyplot.legend()
pyplot.grid()
pyplot.show()