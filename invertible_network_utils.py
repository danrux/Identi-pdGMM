"""Create invertible mixing networks."""

import numpy as np
import torch
import os



# this function is modified based on https://github.com/slachapelle/disentanglement_via_mechanism_sparsity
def get_decoder(x_dim, z_dim, seed, n_layers, load_f, load_slopes, save_dir, manifold='nn', smooth=False):
    rng_data_gen = np.random.default_rng(seed)

    if manifold == "nn":

        # NOTE: injectivity requires z_dim <= h_dim <= x_dim
        h_dim = x_dim
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if load_f is not None:
            dd = np.load(load_f)
            W0 = [dd[f] for f in dd]
            neg_slopes = np.loadtxt(load_slopes, delimiter=',')

        else:
            W0 = []
            neg_slopes = np.random.uniform(0.5, 1.5, n_layers)
            W1 = rng_data_gen.normal(size=(z_dim, h_dim))
            W1 = np.linalg.qr(W1.T)[0].T
            W0.append(W1)
            for l in range(n_layers-1):

                Wi = rng_data_gen.normal(size=(h_dim, x_dim))
                Wi = np.linalg.qr(Wi.T)[0].T
                W0.append(Wi)
            if save_dir is not None:
                save_path_f = os.path.join(save_dir, 'f.npz')
                save_path_slope = os.path.join(save_dir, 'slopes.csv')
                np.savez(save_path_f, *W0)
                np.savetxt(save_path_slope, neg_slopes, delimiter=',')

        W = []
        for l in range(n_layers):
            Wi = W0[l]
            Wi = torch.Tensor(Wi).to(device)
            Wi.requires_grad = False
            W.append(Wi)

        # note that this decoder is almost surely invertible WHEN dim <= h_dim <= x_dim
        # since Wx is injective
        # when columns are linearly indep, which happens almost surely,
        # plus, composition of injective functions is injective.
        def decoder(z):
            with torch.no_grad():

                z = torch.Tensor(z).to(device)
                h = torch.matmul(z, W[0])
                if n_layers > 1:
                    for l in range(n_layers - 1):
                        neg_slope = neg_slopes[l+1]
                        if smooth:
                            h = neg_slope * h + (1 - neg_slope) * torch.log(1 + torch.exp(h))
                        else:
                            h = torch.maximum(neg_slope * h, h)  # leaky relu

                        h = torch.matmul(h, W[l + 1])

            return h

    else:
        raise NotImplementedError(f"The manifold {manifold} is not implemented.")

    return decoder