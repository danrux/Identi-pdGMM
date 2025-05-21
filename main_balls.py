import numpy as np
import argparse
import random
import os
import csv
from torch import nn
from evaluation import MCC
import cooper
from generate_balls_dataset import generate_ball_dataset
import encoders
import torch
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as PLT
import torch.distributions as D
import disentanglement_utils
from sklearn.preprocessing import StandardScaler

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = "cuda"
else:
    device = "cpu"

print("device:", device)


def main():
    args, parser = parse_args()

    if not args.evaluate_redu:
        args.save_dir = os.path.join(args.model_dir, f'balls{args.n_balls}_rs{args.seed}_prob{args.mask_prob}')

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        results_file = os.path.join(args.save_dir, 'results.csv')
    else:
        args.load_redu = os.path.join(args.model_dir, f'balls{args.n_balls}_rs{args.seed}_prob{args.mask_prob}',
                                      'linearredu.pth')
        args.load_lambdas = os.path.join(args.model_dir, f'balls{args.n_balls}_rs{args.seed}_prob{args.mask_prob}',
                                         'lambdas.pth')

    if not args.evaluate_cnn:
        args.save_dir = os.path.join(args.model_dir, f'balls{args.n_balls}_rs{args.seed}_prob{args.mask_prob}')

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        results_file = os.path.join(args.save_dir, 'results.csv')
    else:
        args.load_cnn = os.path.join(args.model_dir, f'balls{args.n_balls}_rs{args.seed}_prob{args.mask_prob}',
                                     'cnn.pth')

    if not args.evaluate:
        args.save_dir = os.path.join(args.model_dir, f'balls{args.n_balls}_rs{args.seed}_prob{args.mask_prob}')

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        results_file = os.path.join(args.save_dir, 'results.csv')
    else:
        args.load_g = os.path.join(args.model_dir, f'balls{args.n_balls}_rs{args.seed}_prob{args.mask_prob}', 'g.pth')
        args.load_f_hat = os.path.join(args.model_dir, f'balls{args.n_balls}_rs{args.seed}_prob{args.mask_prob}',
                                       'f_hat.pth')
        args.n_steps = 1

    global device
    if args.no_cuda:
        device = "cpu"
        print("Using cpu")
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)


    class ConvAutoencoder(nn.Module):
        def __init__(self):
            super(ConvAutoencoder, self).__init__()

            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(4, 4),

            )

            # Decoder
            self.decoder = nn.Sequential(

                nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 3, kernel_size=4, stride=4),
                nn.Sigmoid()  # Output values normalized to [0, 1]
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    # Initialize the model, loss function, and optimizer
    cnn = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()  # Mean Squared Error for reconstruction
    optimizer = optim.Adam(cnn.parameters(), lr=args.lr_cnn)

    def generate_points(num_points=args.n_balls, min_val=0.3, max_val=0.7, min_distance=args.dist):
        while True:
            # Generate all points at once
            points = np.random.uniform(min_val, max_val, size=(num_points, 2))

            # Check pairwise distances
            distances = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)
            np.fill_diagonal(distances, np.inf)  # Ignore self-distances

            # Ensure all distances are greater than the minimum distance
            if np.all(distances > min_distance):
                return points

    # Generate the means for each ball
    means = generate_points()
    args.mask_values = means

    # Training function
    def train_autoencoder(model, criterion, optimizer, num_epochs=20):
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            data_z, data = generate_ball_dataset(n_balls=args.n_balls, means=means, mask_values=args.mask_values,
                                                 Sigma=args.Sigma,
                                                 mask_prob=args.mask_prob,
                                                 sample_num=args.batch_size)

            data = (data / 255).to(device)

            # Forward pass
            reconstructed = model(data)
            loss = criterion(reconstructed, data)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            balls = (data * 255).cpu().detach().numpy()
            balls1 = (reconstructed * 255).cpu().detach().numpy()

            if epoch % 250 == 1:
                print((model.encoder(data)).size())
                save_path = os.path.join(args.save_dir, 'cnn.pth')
                torch.save(model.state_dict(), save_path)
                PLT.imshow((np.transpose(np.array(balls[0, :, :, :], dtype='uint8'), axes=(1, 2, 0))))
                PLT.show()
                PLT.imshow((np.transpose(np.array(balls[122, :, :, :], dtype='uint8'), axes=(1, 2, 0))))
                PLT.show()
                PLT.imshow((np.transpose(np.array(balls[166, :, :, :], dtype='uint8'), axes=(1, 2, 0))))
                PLT.show()
                PLT.imshow((np.transpose(np.array(balls1[0, :, :, :], dtype='uint8'), axes=(1, 2, 0))))
                PLT.show()
                PLT.imshow((np.transpose(np.array(balls1[122, :, :, :], dtype='uint8'), axes=(1, 2, 0))))
                PLT.show()
                PLT.imshow((np.transpose(np.array(balls1[166, :, :, :], dtype='uint8'), axes=(1, 2, 0))))
                PLT.show()

                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss}")

    if args.load_cnn is not None:
        cnn.load_state_dict(torch.load(args.load_cnn, map_location=torch.device(device)))
    if not args.evaluate_cnn:
        train_autoencoder(cnn, criterion, optimizer, num_epochs=args.n_steps_cnn)
    else:
        cnn.eval()


    ### second phase: stage 1
    # Define the Convolutional Autoencoder model for 3-channel data
    class LinearRedu(nn.Module):
        def __init__(self):
            super(LinearRedu, self).__init__()

            # Encoder
            self.encoder = encoders.get_mlp(
                n_in=32*4,
                n_out=args.n_balls * 2,
                layers=[

                    (args.n_balls) * 50,
                    (args.n_balls) * 100,
                    (args.n_balls) * 100,
                    (args.n_balls) * 50,

                ],
                output_normalization="bn",
                # linear=True
            )

            # Decoder
            self.decoder = encoders.get_mlp(
                n_in=args.n_balls * 2,
                n_out=32*4,
                layers=[

                    (args.n_balls) * 50,
                    (args.n_balls) * 100,
                    (args.n_balls) * 100,
                    (args.n_balls) * 50,

                ],
                # output_normalization="bn",
                # linear=True
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    data_z, data_graph = generate_ball_dataset(n_balls=args.n_balls, means=means, mask_values=args.mask_values,
                                               Sigma=args.Sigma,
                                               mask_prob=args.mask_prob,
                                               sample_num=args.batch_size)

    data_graph = (data_graph / 255).to(device)

    # Forward pass
    data = (cnn.encoder(data_graph)).squeeze(2).squeeze(2)

    data = torch.tensor(data, requires_grad=False).to(device)
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True)

    criterion = nn.MSELoss()  # Mean Squared Error for reconstruction
    linearredu = LinearRedu().to(device)

    optimizer = optim.Adam(linearredu.parameters(), lr=args.lr_redu_linear)

    def train_linearredu(model, criterion, optimizer, num_epochs=20):
        model.train()

        for epoch in range(num_epochs):
            total_loss = 0
            data_z, data_graph = generate_ball_dataset(n_balls=args.n_balls, means=means, mask_values=args.mask_values,
                                                       Sigma=args.Sigma,
                                                       mask_prob=args.mask_prob,
                                                       sample_num=args.batch_size)

            data_graph = (data_graph / 255).to(device)

            # Forward pass
            data = (cnn.encoder(data_graph)).squeeze(2).squeeze(2)

            data = data.clone().detach().requires_grad_(True).to(device)
            data = (data - mean)

            # Forward pass
            reconstructed = model(data)
            z_hat = model.encoder(data)

            loss_rec = criterion(reconstructed, data)

            # dimension alignment
            mvn = D.multivariate_normal.MultivariateNormal(torch.zeros(args.n_balls * 2).to(device),
                                                           torch.eye(args.n_balls * 2).to(device))
            loss_prior = mvn.log_prob(z_hat).mean()
            loss = loss_rec - loss_prior


            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            balls = (data_graph * 255).cpu().detach().numpy()
            balls0 = (cnn(data_graph) * 255).cpu().detach().numpy()
            balls1 = (cnn.decoder((reconstructed + mean).unsqueeze(2).unsqueeze(3)) * 255).cpu().detach().numpy()

            if epoch % 250 == 1:
                print('loss_rec', loss_rec)
                print('loss_prior', loss_prior)

                save_path = os.path.join(args.save_dir, 'linearredu.pth')
                torch.save(model.state_dict(), save_path)

                PLT.imshow((np.transpose(np.array(balls0[0, :, :, :], dtype='uint8'), axes=(1, 2, 0))))
                PLT.show()
                PLT.imshow((np.transpose(np.array(balls0[122, :, :, :], dtype='uint8'), axes=(1, 2, 0))))
                PLT.show()

                PLT.imshow((np.transpose(np.array(balls1[0, :, :, :], dtype='uint8'), axes=(1, 2, 0))))
                PLT.show()
                PLT.imshow((np.transpose(np.array(balls1[122, :, :, :], dtype='uint8'), axes=(1, 2, 0))))
                PLT.show()

                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss}")

    if args.load_redu is not None:
        linearredu.load_state_dict(torch.load(args.load_redu, map_location=torch.device(device)))

    if not args.evaluate_redu:
        train_linearredu(linearredu, criterion, optimizer, num_epochs=args.n_steps_redulinear)
    else:
        linearredu.eval()

    ### test second phase
    z_true, data_graph = generate_ball_dataset(n_balls=args.n_balls, means=means, mask_values=args.mask_values,
                                               Sigma=args.Sigma,
                                               mask_prob=args.mask_prob,
                                               sample_num=args.batch_size * 5)

    data_graph = (data_graph / 255).to(device)

    # Forward pass
    data = (cnn.encoder(data_graph)).squeeze(2).squeeze(2)

    data = data.clone().detach().requires_grad_(True).to(device)
    z_data = linearredu.encoder(data - mean)

    for i in range(1):

        x_train = z_true

        scaler_x = StandardScaler()
        x_train = scaler_x.fit_transform(x_train.detach().cpu().numpy())

        for j in range(args.n_balls * 2):
            y_train = z_data[:, j].reshape((-1, 1))
            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train.detach().cpu().numpy())
            linear_model = disentanglement_utils.linear_disentanglement(y_train, x_train, train_mode=True)
            (
                linear_disentanglement_score,
                _,
            ), _ = disentanglement_utils.linear_disentanglement(
                y_train, x_train, mode="r2", model=linear_model
            )

            print('group:', i, 'ground truth:', j)
            print(linear_disentanglement_score)

    ########## third phase: stage 2 with sparsity constraint
    class Constrained_DE(cooper.ConstrainedMinimizationProblem):
        def __init__(self):
            self.criterion = nn.MSELoss(reduction='mean')

            super().__init__(is_constrained=True)

        def closure(self, inputs):
            x = inputs
            z_hat = g(x)
            x_hat = f_hat(z_hat)

            loss = self.criterion(x_hat, x)
            ineq_defect = torch.sum(torch.abs(z_hat)) / args.batch_size / args.n_balls  - args.sparse_level

            return cooper.CMPState(loss=loss, ineq_defect=ineq_defect, eq_defect=None)

    g = encoders.get_mlp(
        n_in=args.n_balls * 2,
        n_out=args.n_balls * 2,
        layers=[

            args.n_balls * 2 * 10,
            args.n_balls * 2 * 50,
            args.n_balls * 2 * 50,
            args.n_balls * 2 * 50,
            args.n_balls * 2 * 50,
            args.n_balls * 2 * 10,

        ],
        output_normalization="bn",
        linear=True
    )

    f_hat = encoders.get_mlp(
        n_in=args.n_balls * 2,
        n_out=args.n_balls * 2,
        layers=[

            args.n_balls * 2 * 10,
            args.n_balls * 2 * 50,
            args.n_balls * 2 * 50,
            args.n_balls * 2 * 50,
            args.n_balls * 2 * 50,
            args.n_balls * 2 * 10,

        ],
        # output_normalization="bn",
        linear=True
    )

    if args.load_g is not None:
        g.load_state_dict(torch.load(args.load_g, map_location=torch.device(device)))


    if args.load_f_hat is not None:
        f_hat.load_state_dict(torch.load(args.load_f_hat, map_location=torch.device(device)))

    g = g.to(device)
    f_hat = f_hat.to(device)

    params = list(g.parameters()) + list(f_hat.parameters())

    ############### constarint optimization ###################
    cmp_vade = Constrained_DE()
    formulation = cooper.LagrangianFormulation(cmp_vade, args.aug_lag_coefficient)
    primal_optimizer = cooper.optim.ExtraAdam(params, lr=args.lr)
    dual_optimizer = cooper.optim.partial_optimizer(cooper.optim.ExtraAdam, lr=args.lr / 2)

    coop_vade = cooper.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optimizer,
        dual_optimizer=dual_optimizer,
    )

    total_loss_values = []
    global_step = len(total_loss_values) + 1

    while (global_step <= args.n_steps):
        if not (args.evaluate):
            g.train()
            f_hat.train()

            data_z, data_graph = generate_ball_dataset(n_balls=args.n_balls, means=means, mask_values=args.mask_values,
                                                       Sigma=args.Sigma,
                                                       mask_prob=args.mask_prob,
                                                       sample_num=args.batch_size)

            data_graph = (data_graph / 255).to(device)

            # Forward pass
            data_cnn = (cnn.encoder(data_graph)).squeeze(2).squeeze(2)


            data = linearredu.encoder(data_cnn - mean)



            data = data.clone().detach().requires_grad_(True).to(device)
            coop_vade.zero_grad()
            lagrangian = formulation.composite_objective(
                cmp_vade.closure, data
            )
            formulation.custom_backward(lagrangian)
            coop_vade.step(cmp_vade.closure, data)

        if global_step % args.n_log_steps == 1 or global_step == args.n_steps:
            f_hat.eval()
            g.eval()
            data_z, data_graph = generate_ball_dataset(n_balls=args.n_balls, means=means, mask_values=args.mask_values,
                                                       Sigma=args.Sigma,
                                                       mask_prob=args.mask_prob,
                                                       sample_num=args.batch_size)

            data_graph = (data_graph / 255).to(device)

            # Forward pass
            data_cnn = (cnn.encoder(data_graph)).squeeze(2).squeeze(2)

            # data_cnn = data_cnn.clone().detach().requires_grad_(True).to(device)
            data = linearredu.encoder(data_cnn - mean)



            hz_disentanglement = g(data)

            mcc, cor_m = MCC(data_z, hz_disentanglement, args.n_balls * 2)
            mcc = mcc / (args.n_balls * 2)
            mind = linear_sum_assignment(-1 * cor_m)[1]

            if not args.evaluate:
                fileobj = open(results_file, 'a+')
                writer = csv.writer(fileobj)
                wri = ['MCC', mcc]
                writer.writerow(wri)
                print(global_step)
                print('estimate_mcc')
                print(mcc)

                save_path = os.path.join(args.save_dir, 'g.pth')
                torch.save(g.state_dict(), save_path)
                save_path = os.path.join(args.save_dir, 'f_hat.pth')
                torch.save(f_hat.state_dict(), save_path)

        global_step += 1
        if mcc > 0.995:
            break

    # mainly for visualization after linear rotation

    z_true, data_graph = generate_ball_dataset(n_balls=args.n_balls, means=means, mask_values=args.mask_values,
                                               Sigma=args.Sigma,
                                               mask_prob=args.mask_prob,
                                               sample_num=args.batch_size * 5)

    data_graph = (data_graph / 255).to(device)
    data_cnn = ((cnn.encoder(data_graph)).squeeze(2).squeeze(2))
    data = linearredu.encoder(data_cnn - mean)

    '''
                    max_indices = torch.argmax(mask_prob, dim=1)
                    for i in range(len(args.masks)):
                        data[max_indices == i] = data[max_indices == i] - debias[i,:]
                    '''

    hz_disentanglement = g(data)
    z_data = g(data)

    mcc, cor_m = MCC(z_true, z_data, args.n_balls * 2)
    print('after linear rotation:', mcc / (args.n_balls * 2))

    for i in range(args.n_balls):

        x_train = z_true[:, 2 * i:2 * (i + 1)]
        scaler_x = StandardScaler()
        x_train = scaler_x.fit_transform(x_train.detach().cpu().numpy())

        for j in range(args.n_balls * 2):
            y_train = z_data[:, j].reshape((-1, 1))
            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train.detach().cpu().numpy())
            linear_model = disentanglement_utils.linear_disentanglement(y_train, x_train, train_mode=True)
            (
                linear_disentanglement_score,
                _,
            ), _ = disentanglement_utils.linear_disentanglement(
                y_train, x_train, mode="r2", model=linear_model
            )

            print('group:', i, 'ground truth:', j)
            print(linear_disentanglement_score)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-balls", type=int, default=2)
    parser.add_argument("--mask_prob", type=float, default=0.1)
    parser.add_argument("--dist", type=float, default=0.2)
    parser.add_argument("--means", type=int, default=None)
    parser.add_argument("--mask-values", type=int, default=None)
    parser.add_argument("--Sigma", type=float, default=[[0.01, 0.00], [0.00, 0.01]])
    parser.add_argument("--evaluate_cnn", action='store_true')
    parser.add_argument("--evaluate-redu", action='store_true')
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--resume", action='store_false')

    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--model-dir", type=str, default="Balls_2025")

    parser.add_argument("--save-dir", type=str, default="")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_cnn", type=float, default=1e-3)
    parser.add_argument("--lr-redu-linear", type=float, default=1e-3)
    parser.add_argument("--no-cuda", action="store_true")

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-log-steps", type=int, default=250)
    parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument("--n-steps-cnn", type=int, default=20000)
    parser.add_argument("--n-steps-redulinear", type=int, default=20000)
    parser.add_argument("--load-g", default=None)
    parser.add_argument("--load-f-hat", default=None)
    parser.add_argument("--load-cnn", default=None)
    parser.add_argument("--load-redu", default=None)
    parser.add_argument("--load-lambdas", default=None)
    parser.add_argument("--aug-lag-coefficient", type=float, default=0.00)
    parser.add_argument("--sparse-level", type=float, default=0.01)

    args = parser.parse_args()

    return args, parser


if __name__ == "__main__":
    main()