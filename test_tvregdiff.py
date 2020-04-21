"""
Unit tests for tvregdiff.py which test if this Python
implementation of TVREGDiff replicates the MATLAB demo
scipt outputs from Rick Chartrand's webpage:
https://sites.google.com/site/dnartrahckcir/home/tvdiff-code
"""

import os
import unittest
import numpy as np
from numpy.testing import assert_allclose
from tvregdiff import TVRegDiff

import torch


class TVDiffTest(unittest.TestCase):

    def setUp(self):
        self.data_dir = 'test_data'

    def test_case_small(self):
        """Small-scale example from paper Rick Chartrand,
        "Numerical differentiation of noisy, nonsmooth data," ISRN
        Applied Mathematics, Vol. 2011, Article ID 164564, 2011.
        """

        # Load data (data is from smalldemodata.mat)
        # noisy_abs_data = np.loadtxt('smalldemodata.csv')
        # self.assertEqual(noisy_abs_data.shape, (100,))

        data = torch.load('my_data/y_true_0.pt')[:, 1].numpy()
        noisy_abs_data = torch.load('my_data/y_true_noise_0.pt')[:, 1].numpy()

        dydt = torch.load('my_data/dydt_true_0.pt')[:, 1].numpy()

        # Test with one iteration
        n_iters = 1000
        alph = 0.2
        scale = 'large'
        ep = 1e-6
        dx = 40/1000
        u = TVRegDiff(data, n_iters, alph, u0=None, scale=scale,
                      ep=ep, dx=dx, plotflag=False, diagflag=True)
        # self.assertEqual(u.shape, (101,))
        filepath = os.path.join(self.data_dir, 'smalldemo_u1.csv')
        u_test = np.loadtxt(filepath)
        # assert_allclose(u, u_test)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(u, label='TVReg')
        plt.plot(dydt, '--', label='Exact')
        plt.plot([20, 20], [-0.2, 0])

        # Test with 500 iterations
        # n_iters = 500
        # u = TVRegDiff(noisy_abs_data, n_iters, alph, u0=None, scale=scale,
        #               ep=ep, dx=dx, plotflag=False, diagflag=True)
        # self.assertEqual(u.shape, (101,))
        # filepath = os.path.join(self.data_dir, 'smalldemo_u.csv')
        # u_test = np.loadtxt(filepath)
        # assert_allclose(u, u_test)


if __name__ == '__main__':
    # unittest.main()

    data = torch.load('my_data/y_true_0.pt')[:, 1].numpy()
    noisy_abs_data = torch.load('my_data/y_true_noise_0.pt')[:, 1].numpy()

    dydt = torch.load('my_data/dydt_true_0.pt')[:, 1].numpy()

    # Test with one iteration
    n_iters = 500
    alph = 0.2
    scale = 'large'
    ep = 1e-6
    dx = 40/999
    u = TVRegDiff(noisy_abs_data, n_iters, alph, u0=None, scale=scale,
                  ep=ep, dx=dx, plotflag=False, diagflag=True)

    import scipy.signal
    u_sav = scipy.signal.savgol_filter(noisy_abs_data, window_length=41, polyorder=2, deriv=1, delta=dx)

    import matplotlib.pyplot as plt
    plt.close('all')
    plt.figure()
    plt.plot(u, label='TVReg')
    plt.plot(u_sav, label='scipy')
    plt.plot(dydt, '--', label='Exact')
    plt.legend()
    #plt.xlim([0,100])
