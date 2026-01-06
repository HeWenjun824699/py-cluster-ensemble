import os
import random
import torch
import torch.nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pickle
from types import SimpleNamespace

from .model import Model
from .utils import AKIData, evaluate, get_embedding


def train_representation(input_path, output_path, input_dim, hidden_dim, **kwargs):
    """
    Receive parameters directly for training; unspecified parameters will use default values.
    """

    # 1. Define default parameter configuration (corresponding to original argparse defaults)
    cfg = {
        # --- Path Parameters ---
        'input_path': input_path,
        'filename_train': 'data_train.pkl',
        'filename_valid': 'data_valid.pkl',
        'filename_test': 'data_valid.pkl',
        'filename_data': 'data.pkl',
        'log_path': output_path + './runs',
        'output_path': output_path + "./representations",

        # --- Training Parameters ---
        'pre_epoch': 1,
        'epoch': 100,
        'batch_size': 32,
        'lr': 1e-4,
        'dropout': 0.0,

        # --- Model Parameters ---
        'input_dim': input_dim,
        'hidden_dims': hidden_dim,
        'n_layers': 1,
        'lambda_AE': 1.0,
        'lambda_outcome': 10.0,
        'seed': 2026,

        # --- Utility Parameters ---
        'cuda': 1 if torch.cuda.is_available() else 0
    }

    # 2. Update configuration with user-provided kwargs (override defaults)
    # Example: passing n_layers=2 will override 'n_layers': 1 above
    cfg.update(kwargs)

    # 3. Convert dictionary to object (Namespace)
    # This way, args.input_dim in model.py works correctly without changing model code
    args = SimpleNamespace(**cfg)

    # ================= Logic below remains unchanged =================

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Path handling
    if not args.input_path.endswith('/'):
        args.input_path += '/'

    path_train = args.input_path + args.filename_train
    path_valid = args.input_path + args.filename_valid
    path_data = args.input_path + args.filename_data
    path_test = args.input_path + args.filename_test

    # Load data
    data_train = AKIData(path_train)
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=False)
    data_valid = AKIData(path_valid)
    dataloader_valid = DataLoader(data_valid, batch_size=args.batch_size, shuffle=True)

    # Create output directory
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    writer = SummaryWriter(args.log_path)

    # Initialize model (Model receives args object)
    model = Model(args)

    # Pretrain
    model.pretrain(dataloader_train, verbose=False)

    saved_iter = -1
    saved_iter_list = []
    min_outcome_likelihood = np.inf
    device = torch.device('cuda' if args.cuda else 'cpu')

    print(f"-> Training | input_dim={args.input_dim} | hidden_dims={args.hidden_dims} | epoch={args.epoch}")

    for e in range(args.epoch):
        train_ae_loss, train_outcome_loss, train_outcome_auc_score = model.fit(dataloader_train, verbose=False)
        test_ae_loss, test_outcome_loss, test_outcome_auc_score = evaluate(model, dataloader_valid)

        writer.add_scalar('train_ae_loss', train_ae_loss, e)
        writer.add_scalar('validation_ae_loss', test_ae_loss, e)
        writer.add_scalar('train_outcome_loss', train_outcome_loss, e)
        writer.add_scalar('validation_outcome_loss', train_outcome_loss, e)
        writer.add_scalar('train_outcome_auc', train_outcome_auc_score, e)
        writer.add_scalar('validation_outcome_auc', test_outcome_auc_score, e)

        if test_outcome_loss < min_outcome_likelihood:
            min_outcome_likelihood = test_outcome_loss
            torch.save(model.state_dict(), os.path.join(args.output_path, f'model_best_{args.hidden_dims}.pt'))
            saved_iter = e
            saved_iter_list.append(saved_iter)

    # Generate final representation
    data = AKIData(path_data)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

    model_best = Model(args)
    model_best.load_state_dict(torch.load(
        os.path.join(args.output_path, f'model_best_{args.hidden_dims}.pt'),
        map_location=device
    ))

    # # test on test set
    # data_test = AKIData(args.input_path + args.filename_test)
    # dataloader_test = DataLoader(data_test, batch_size=16, shuffle=False)
    # test_ae_loss, test_outcome_loss, test_outcome_auc_score = evaluate(model_best, dataloader_test)
    # print('test_ae_loss: %.3f' % test_ae_loss)
    # print('test_outcome_loss: %.3f' % test_outcome_loss)
    # print('test_outcome_auc_score: %.3f' % test_outcome_auc_score)

    rep = get_embedding(model_best, dataloader)
    with open(os.path.join(args.output_path, f'rep_{args.hidden_dims}.pkl'), 'wb') as f:
        pickle.dump(rep, f)

    print(f"-> Done. Representation saved: rep_{args.hidden_dims}.pkl")
    writer.close()
    f.close()
