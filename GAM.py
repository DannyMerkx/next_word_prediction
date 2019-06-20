#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:37:46 2019

@author: danny
"""

import pygam
import matplotlib.pyplot as plt
import numpy as np

X1 = np.random.rand(1000)
X2 = np.random.binomial(1,0.5,1000)
X = np.stack([X1,X2]).T
y = np.random.rand(1000)

model = pygam.GAM(pygam.s(0) + pygam.te(0,1), distribution = 'normal')
model.fit(X, y)

for i, term in enumerate(model.terms):
    if term.isintercept:
        continue

    XX = model.generate_X_grid(term=i)
    pdep, confi = model.partial_dependence(term=i, X=XX, width=0.95)

    plt.figure()
    plt.plot(XX[:, term.feature], pdep)
    plt.plot(XX[:, term.feature], confi, c='r', ls='--')
    plt.title(repr(term))
    plt.show()

