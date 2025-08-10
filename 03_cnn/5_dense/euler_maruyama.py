import numpy
from matplotlib import pyplot

def simulate_gbm(mu=0.1, sigma=0.2, s0=1.0, T=1.0, N=1000, seed=1):
    rng = numpy.random.default_rng(seed)
    dt = T / N
    sqrt_dt = numpy.sqrt(dt)
    t = numpy.linspace(0, T, N+1)
    S = numpy.empty(N+1)
    S[0] = s0
    for i in range(N):
        dW = sqrt_dt * rng.standard_normal()
        S[i+1] = S[i] + mu * S[i] * dt + sigma * S[i] * dW
    return t, S

t, S = simulate_gbm()


pyplot.title('Geometric Brownian Motion (Euler-Maruyama)')
pyplot.plot(t, S)
pyplot.xlabel('Time')
pyplot.ylabel('S(t)')
pyplot.grid()
pyplot.show()
