#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 16:15:36 2025

@author: zjpeters
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math


# plot standard normal distribution
mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.title('Standard normal distribution with $\mu=0$, $\sigma^2=1$')
plt.show()

#%% plot gamma distribution
plt.close('all')
x = np.linspace(0,20,100)
plt.plot(x, stats.gamma.pdf(x, a=1, scale=2), label=r'$\alpha=1.00,\theta=2.00$')
plt.plot(x, stats.gamma.pdf(x, a=2, scale=2), label=r'$\alpha=2.00,\theta=2.00$')
plt.plot(x, stats.gamma.pdf(x, a=3, scale=2), label=r'$\alpha=3.00,\theta=2.00$')
plt.plot(x, stats.gamma.pdf(x, a=5, scale=1), label=r'$\alpha=5.00,\theta=1.00$')
plt.plot(x, stats.gamma.pdf(x, a=9, scale=0.5), label=r'$\alpha=9.00,\theta=0.50$')
plt.plot(x, stats.gamma.pdf(x, a=7.5, scale=1), label=r'$\alpha=7.50,\theta=1.00$')
plt.plot(x, stats.gamma.pdf(x, a=0.5, scale=1), label=r'$\alpha=0.50,\theta=1.00$')
plt.title(r'Gamma distribution using different $\alpha$ and $\theta$ values')
plt.legend()

#%% plot Poisson distribution
K = np.arange(21)
plt.close('all')
l=1
y = []
for k in K:
    y.append((l**k*np.exp(-l))/math.factorial(k))
plt.scatter(K,y)
plt.plot(K,y, label=r'$\lambda=1$')
plt.legend()
l=4
y = []
for k in K:
    y.append((l**k*np.exp(-l))/math.factorial(k))
plt.scatter(K,y)
plt.plot(K,y, label=r'$\lambda=4$')
plt.legend()
l=10
y = []
for k in K:
    y.append((l**k*np.exp(-l))/math.factorial(k))
plt.scatter(K,y)
plt.plot(K,y, label=r'$\lambda=10$')
plt.legend()
plt.title(r'Poisson distribution using different $\lambda$')

#%% make linear regression lines
plt.close('all')
def MakeObservations(N):       # N number of data pairs
    Nrange = 20;               # set scale
    SigmaY = 200;
    X  = np.random.uniform(0,Nrange+1,N);      # uniformly distributed between 0 and 20
    Ymax = 0.5 * Nrange**3;                    # upper bound
    MuY = 0.5 * X**3 - 0.5 * Ymax             # non-linear but monotonic dependence, between 0 and 200
    Y  = np.random.normal(MuY, SigmaY );       # binomially distributed around mean R1
    Noutlier = np.floor(N/5).astype(int)       #  add 20% outliers
    ix = np.random.permutation( np.arange(0,N));
    Y[ ix[0:Noutlier] ] = np.random.uniform(low=-Ymax, high=Ymax, size=Noutlier)
    return X, Y;

S = 50
x_i,y_i = MakeObservations(S)

# linear fit
n = 1
S_x = 0
S_y = 0

for i in range(S):
    S_x += x_i[i]**n
    S_y += y_i[i]

t_i = x_i**n - (S_x/S)
S_tt = 0
S_ty = 0
for i in range(S):
    S_tt += t_i[i]**2
    S_ty += t_i[i]*y_i[i]

b_hat = S_ty/S_tt
a_hat = (S_y - (S_x*b_hat))/S

sigma_a_2 = (1 + (S_x**2/(S*S_tt)))/S
sigma_b_2 = 1/S_tt
cov_ab = -S_x/(S*S_tt)

x = np.sort(x_i)
y = a_hat + b_hat*x**n
plt.scatter(x_i, y_i, c='tab:blue', label='$x_i$, $y_i$')
plt.plot(x,y, c='tab:orange', linestyle='--', label='Linear fit')

sigma_yi_2 = np.sum((y - np.mean(y_i))**2)/(S-1)
print(f"linear sigma^2_a = {np.sqrt(sigma_a_2)}\n linear sigma^2_b = {np.sqrt(sigma_b_2)}\n linear sigma^2_yi =  {np.sqrt(sigma_yi_2)}")
# polynomial fit
n = 2
S_x = 0
S_y = 0

for i in range(S):
    S_x += x_i[i]**n
    S_y += y_i[i]

t_i = x_i**n - (S_x/S)
S_tt = 0
S_ty = 0
for i in range(S):
    S_tt += t_i[i]**2
    S_ty += t_i[i]*y_i[i]

b_hat = S_ty/S_tt
a_hat = (S_y - (S_x*b_hat))/S

sigma_a_2 = (1 + (S_x**2/(S*S_tt)))/S
sigma_b_2 = 1/S_tt
cov_ab = -S_x/(S*S_tt)

x = np.sort(x_i)
y = a_hat + b_hat*x**n
plt.plot(x,y, c='tab:green', label='Polynomial fit')
plt.xlabel("$x$")
plt.ylabel("$y$")
# plt.axvline(c='k')
# plt.axhline(c='k')
plt.legend()
plt.title("Comparison of fit curves")
sigma_yi_2 = np.sum((y - np.mean(y_i))**2)/(S-1)
