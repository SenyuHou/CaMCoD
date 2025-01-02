import tqdm
import random
import torch
import argparse
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.utils.data as data
import torchvision
from utils.ema import EMA
from utils.vit_wrapper import vit_img_wrap
from utils.cifar_data_utils import *
from utils.model_ResNet import ResNet_encoder
from utils.model_SimCLR import SimCLR_encoder
from utils.plot_loss import plot_and_save_losses
import torch.optim as optim
from utils.learning import *
from utils.noisy_label_purified import *
from utils.ws_augmentation import *
from utils.model_diffusion import Diffusion
from utils.add_noise import *
from utils.create_imbalanced_data import *
from utils.log_config import setup_logger

# Main training function (diffusion model, training, validation, test sets, model save path, command line arguments, encoder)
def train(diffusion_model1, diffusion_model2, train_dataset, test_dataset, model_path, args, vit_fp, fp_dim):
    """
    Train the diffusion model with the given datasets and arguments.

    Parameters:
    - diffusion_model: The diffusion model to be trained.
    - train_dataset: The dataset used for training.
    - test_dataset: The dataset used for testing.
    - val_dataset: The dataset used for validation.
    - model_path: Path to save the trained model.
    - args: Command line arguments containing training parameters.
    - vit_fp: Whether to use precomputed feature embeddings.
    - fp_dim: Dimension of the feature embeddings.
    """
    # Extract configurations from the model and command line arguments, including training device, number of classes, total training epochs, k value in KNN, and warmup epochs.
    device = diffusion_model1.device
    n_class = diffusion_model1.n_class
    n_epochs = args.nepoch
    warmup_epochs = args.warmup_epochs
    noise_class = args.noise_type.split('-')[1]
    noise_ratio = float(args.noise_type.split('-')[2])
    seed = args.seed
    
    if noise_class == 'sym':
        noisy_targets = add_noise(train_dataset.targets, noise_ratio, n_class, seed, transition=None, symmetric_noise=True)
        train_dataset.update_label(noisy_targets)
        noisy_labels = torch.tensor(train_dataset.targets).squeeze().to(device)
        print(f'Training on {args.noise_type} label noise:')
    elif noise_class == 'asym':
        transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}
        noisy_targets = add_noise(train_dataset.targets, noise_ratio, n_class, seed, transition, symmetric_noise=False)
        train_dataset.update_label(noisy_targets)
        noisy_labels = torch.tensor(train_dataset.targets).squeeze().to(device)
        print(f'Training on {args.noise_type} label noise:')
    else:
        print("Check your noise type carefully!")

    # Compute embedding fp(x) for ws_dataset
    dataset = args.noise_type.split('-')[0]
    if dataset == 'cifar10':
        data_dir  = os.path.join(os.getcwd(), './data/cifar-10-batches-py')
    else:
        data_dir  = os.path.join(os.getcwd(), './data/cifar-100-python')
    print('Doing pre-computing fp embeddings for weak and strong dataset')
    train_embed_dir = os.path.join(data_dir, f'fp_embed_imbalanced_cifar')
    test_embed_dir = os.path.join(data_dir, f'fp_embed_test_cifar.npy')
    weak_embed, strong_embed = prepare_2_fp_x(diffusion_model1.fp_encoder, train_dataset, save_dir=train_embed_dir, device=device, fp_dim=fp_dim)
    print('Doing pre-computing fp embeddings for test dataset')
    test_embed = prepare_fp_x(diffusion_model1.fp_encoder, test_dataset, save_dir=test_embed_dir, device=device, fp_dim=fp_dim)
    weak_embed = weak_embed.to(device)
    strong_embed = strong_embed.to(device)
    test_embed = test_embed.to(device)
    
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=200, shuffle=False, num_workers=4)

    # Optimizer settings
    optimizer1 = optim.Adam(diffusion_model1.model.parameters(), lr=0.0001, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)
    optimizer2 = optim.Adam(diffusion_model1.model.parameters(), lr=0.0001, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)
    diffusion_loss = nn.MSELoss(reduction='none')

    # Initialize EMA helper and register model parameters to smooth model parameter updates during training to improve model stability and performance.
    ema_helper1 = EMA(mu=0.9999)
    ema_helper1.register(diffusion_model1.model)
    ema_helper2 = EMA(mu=0.9999)
    ema_helper2.register(diffusion_model2.model)

    # Train in a loop and record the highest accuracy to save the model
    max_accuracy = 0.0
    print('Diffusion training start')

    for epoch in range(n_epochs):
        diffusion_model1.model.train()
        diffusion_model2.model.train()
        
        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'train diffusion epoch {epoch}', ncols=120) as pbar:
            for i, data_batch in pbar:
                [x_batch_w, x_batch_s, y_batch, data_indices] = data_batch[:4]
                x_batch_w = x_batch_w.to(device)
                x_batch_s = x_batch_s.to(device)
                y_noisy = y_batch.to(device)

                # Use precomputed feature embeddings
                fp_embd_w = weak_embed[data_indices, :].to(device)
                fp_embd_s = strong_embed[data_indices, :].to(device)
                
                # Check if the labels are one-hot encoded, if not, convert them to vectors
                y_label_batch_w, y_label_batch_s, weights_factor_w, weights_factor_s = cast_label_to_one_hot_and_prototype(fp_embd_w=fp_embd_w, 
                fp_embd_s=fp_embd_s, 
                y_noisy=y_noisy, 
                weak_embed=weak_embed, 
                strong_embed=strong_embed, 
                noisy_labels=noisy_labels, 
                device=device, n_class=n_class)
                y_0_batch_w = y_label_batch_w.to(device)
                y_0_batch_s = y_label_batch_s.to(device)
                y_noisy_batch = cast_label_to_one_hot(y_noisy.to(torch.int64), n_class=n_class).to(device)

                # Adjust the learning rate
                adjust_learning_rate(optimizer1, i / len(train_loader) + epoch, warmup_epochs=warmup_epochs, n_epochs=200, lr_input=0.001)
                adjust_learning_rate(optimizer2, i / len(train_loader) + epoch, warmup_epochs=warmup_epochs, n_epochs=200, lr_input=0.001)

                # Sampling t1 and t2
                n = x_batch_w.size(0)
                t = torch.randint(low=0, high=diffusion_model1.num_timesteps, size=(n // 2 + 1, )).to(device)
                t = torch.cat([t, diffusion_model1.num_timesteps - 1 - t], dim=0)[:n]
                t1 = t2 = t
                e = torch.randn_like(y_0_batch_w).to(y_0_batch_w.device)
                e_w = e_s = e
                output_s, e_s = diffusion_model1.forward_t(y_0_batch_w, x_batch_w, t1, fp_embd_w, noise=e_s)               
                output_w, e_w = diffusion_model2.forward_t(y_0_batch_s, x_batch_s, t2, fp_embd_s, noise=e_w)

                # Compute loss
                mse_loss_w = diffusion_loss(e_w, output_w)
                mse_loss_s = diffusion_loss(e_s, output_s)

                if epoch >= warmup_epochs:                
                    with torch.no_grad():
                        diffusion_model1.ddim_num_steps = diffusion_model2.ddim_num_steps = 2
                        direct_y_0_w = diffusion_model1.reverse_ddim(x_batch_w, stochastic=False, fp_x = fp_embd_w, fq_x=None, direct_y0=True)               
                        direct_y_0_s = diffusion_model2.reverse_ddim(x_batch_s, stochastic=False, fp_x = fp_embd_s, fq_x=None, direct_y0=True)
                        
                        d_y_0_w = torch.softmax(-(direct_y_0_w.clone() - 1) ** 2, dim=-1)
                        d_y_0_s = torch.softmax(-(direct_y_0_s.clone() - 1) ** 2, dim=-1)
                        
                        GU_w = calculate_entropy(d_y_0_w)  
                        GU_s = calculate_entropy(d_y_0_s)  

                        GI = js_divergence(output_w, output_s)
                        
                        low_prob_w = fit_2dgmm_in_kmeans_and_get_low_mean_prob(y_noisy_batch.cpu().numpy(), GU_w.cpu().numpy(), GI.cpu().numpy())  
                        low_prob_s = fit_2dgmm_in_kmeans_and_get_low_mean_prob(y_noisy_batch.cpu().numpy(), GU_s.cpu().numpy(), GI.cpu().numpy())  
                        low_prob_w = torch.tensor(low_prob_w).to(device).unsqueeze(1)
                        low_prob_s = torch.tensor(low_prob_s).to(device).unsqueeze(1)
                        
                        weight_factor_w = (weights_factor_w).float()
                        weight_factor_s = (weights_factor_s).float()

                    # Compute weighted losses and Mutual guidance
                    mse_loss_w_per_sample = torch.matmul(weight_factor_w, mse_loss_w)
                    mse_loss_s_per_sample = torch.matmul(weight_factor_s, mse_loss_s)
                    loss_w = torch.mean(mse_loss_s_per_sample)
                    loss_s = torch.mean(mse_loss_w_per_sample)

                else:
                    loss_s = torch.mean(mse_loss_w)
                    loss_w = torch.mean(mse_loss_s)
                
                pbar.set_postfix({'loss_w': loss_w.item(), 'loss_s': loss_s.item()})

                optimizer1.zero_grad()
                loss_w.backward()
                torch.nn.utils.clip_grad_norm_(diffusion_model1.model.parameters(), 1.0)
                optimizer1.step()  # Update model 1
                ema_helper1.update(diffusion_model1.model)

                optimizer2.zero_grad()
                loss_s.backward()
                torch.nn.utils.clip_grad_norm_(diffusion_model2.model.parameters(), 1.0)
                optimizer2.step()  # Update model 2
                ema_helper2.update(diffusion_model2.model)

        # Every epoch, perform validation, if the validation accuracy of the current epoch exceeds the previous highest accuracy, evaluate the model on the test set, and save the current best model parameters.
        if epoch >= warmup_epochs:
            test_acc = test(diffusion_model1, test_loader, test_embed, args.ddim_n_step, device)
            logger.info(f"epoch: {epoch}, test accuracy: {test_acc:.2f}%")
            if test_acc > max_accuracy:
                # Save diffusion model
                print('Improved! Evaluate on testing set...')
                states = [
                    diffusion_model1.model.state_dict(),
                    diffusion_model1.diffusion_encoder.state_dict(),
                    diffusion_model1.fp_encoder.state_dict()
                ]
                torch.save(states, model_path)
                message = (f"Model saved, update best accuracy at Epoch {epoch}, test acc: {test_acc:.2f}")
                logger.info(message)
                max_accuracy = max(max_accuracy, test_acc)
        

def test(diffusion_model, test_loader, test_embed, ddim_n_steps, device):
    with torch.no_grad():
        diffusion_model.model.eval()
        diffusion_model.fp_encoder.eval()
        correct_cnt = torch.tensor(0, dtype=torch.long, device=device)  # Move to GPU
        all_cnt = torch.tensor(0, dtype=torch.long, device=device)  # Move to GPU

        for idx, data_batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f'Doing DDIM...', ncols=100):
            images, target, data_indices = data_batch[:3]
            images = images.to(device)
            target = target.to(device)
            fp_test_embed = test_embed[data_indices, :].to(device)
            # Perform reverse DDIM and move result to CPU for comparison
            diffusion_model1.ddim_num_steps = diffusion_model2.ddim_num_steps = ddim_n_steps
            pred_y_0 = diffusion_model.reverse_ddim(images, stochastic=False, fp_x=fp_test_embed, fq_x=None, direct_y0=False).detach().to(device)
            # Calculate correct predictions
            correct = cnt_agree(pred_y_0, target)
            correct_cnt += correct
            all_cnt += images.shape[0]
        # Calculate accuracy on GPU
        acc = 100 * correct_cnt.float() / all_cnt.float()  # Use float to avoid integer division
    return acc.item()  # Use item() to get the Python number from the tensor

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123, help='Random seed for reproducibility')
    # Training parameters
    parser.add_argument('--noise_type', default='cifar10-sym-0.4', help='noise label file', type=str)
    parser.add_argument("--nepoch", default=200, help="number of training epochs", type=int)
    parser.add_argument("--batch_size", default=200, help="batch_size", type=int)
    parser.add_argument("--device", default='cpu', help="which GPU to use", type=str)
    parser.add_argument("--num_workers", default=4, help="num_workers", type=int)
    parser.add_argument("--warmup_epochs", default=5, help="warmup_epochs", type=int)
    # Diffusion model hyperparameters
    parser.add_argument("--feature_dim", default=512, help="feature_dim", type=int)
    parser.add_argument("--ddim_n_step", default=10, help="number of steps in ddim", type=int)
    parser.add_argument("--diff_encoder", default='resnet34', help="which encoder for diffusion (linear, resnet18, 34, 50...)", type=str)
    # Large model hyperparameters
    parser.add_argument("--fp_encoder", default='ViT', help="which encoder for fp (SimCLR, Vit or ResNet)", type=str)
    parser.add_argument("--ViT_type", default='ViT-L/14', help="which encoder for Vit", type=str)
    parser.add_argument("--ResNet_type", default='resnet152', help="which encoder for ResNet", type=str)
    # Storage path
    parser.add_argument("--log_name", default='cifar10-idn-0.1.log', help="create your logs name", type=str)
    args = parser.parse_args()
    logger = setup_logger(args)
    set_random_seed(args.seed)

    # Set GPU or CPU for training
    device = args.device
    print('Using device:', device)

    dataset = args.noise_type.split('-')[0]

    # Load dataset
    if dataset == 'cifar10':
        n_class = 10
        train_dataset_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        test_dataset_cifar = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
        # Data normalization parameters
        CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR_STD = (0.2023, 0.1994, 0.2010)
    elif dataset == 'cifar100':
        n_class = 100
        train_dataset_cifar = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
        test_dataset_cifar = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)
        CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR_STD = (0.2023, 0.1994, 0.2010)
    else:
        raise Exception("Dataset should be cifar10 or cifar100")

    # Load f_phi feature extractor
    if args.fp_encoder == 'ViT':
        fp_encoder1 = vit_img_wrap(args.ViT_type, args.device, center=CIFAR_MEAN, std=CIFAR_STD)
        fp_encoder2= vit_img_wrap(args.ViT_type, args.device, center=CIFAR_MEAN, std=CIFAR_STD)
        fp_dim = fp_encoder1.dim
    elif args.fp_encoder == 'ResNet':
        if args.ResNet_type == 'resnet34':
            fp_dim = 512
        else:
            fp_dim = 2048
        fp_encoder = ResNet_encoder(feature_dim=fp_dim, base_model=args.ResNet_type).to(args.device)
    else:
        raise Exception("fp_encoder should be ResNet or Vit")

    # Create training and test set instances using custom dataset class
    transform_fixmatch = TransformFixMatch_CIFAR10(CIFAR_MEAN, CIFAR_STD, 2, 5)
    train_dataset = Double_dataset(data=train_dataset_cifar.data[:], targets=train_dataset_cifar.targets[:], transform_fixmatch=transform_fixmatch)
    test_dataset = Custom_dataset(test_dataset_cifar.data, test_dataset_cifar.targets)

    # Initialize the diffusion model
    model_path = f'./model/CaMCoD_{args.fp_encoder}_{args.noise_type}.pt'
    diffusion_model1 = Diffusion(fp_encoder=fp_encoder1, n_class=n_class, fp_dim=fp_dim, feature_dim=args.feature_dim, device=device, encoder_type=args.diff_encoder, ddim_num_steps=args.ddim_n_step, beta_schedule='cosine', guidance = True)
    diffusion_model2 = Diffusion(fp_encoder=fp_encoder2, n_class=n_class, fp_dim=fp_dim, feature_dim=args.feature_dim, device=device, encoder_type=args.diff_encoder, ddim_num_steps=args.ddim_n_step, beta_schedule='cosine', guidance = True)
    diffusion_model1.fp_encoder.eval()
    diffusion_model2.fp_encoder.eval()
    # state_dict = torch.load(model_path, map_location=torch.device(device))
    # diffusion_model.load_diffusion_net(state_dict)

    # Train the diffusion model
    print(f'Training CaMCoD using fp encoder: {args.fp_encoder} on: {args.noise_type}.')
    print(f'Model saving dir: {model_path}')
    train(diffusion_model1, diffusion_model2, train_dataset=train_dataset, test_dataset=test_dataset, model_path=model_path, args=args, vit_fp=True, fp_dim=fp_dim)
