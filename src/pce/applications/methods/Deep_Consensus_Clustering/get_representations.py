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
    直接接收参数进行训练，未提供的参数将使用默认值。
    """

    # 1. 定义默认参数配置 (对应原 argparse 的 default)
    cfg = {
        # --- 路径参数 ---
        'input_path': input_path,
        'filename_train': 'data_train.pkl',
        'filename_valid': 'data_valid.pkl',
        'filename_test': 'data_valid.pkl',
        'filename_data': 'data.pkl',
        'log_path': output_path + './runs',
        'output_path': output_path + "./representations",

        # --- 训练参数 ---
        'pre_epoch': 1,
        'epoch': 100,
        'batch_size': 32,
        'lr': 1e-4,
        'dropout': 0.0,

        # --- 模型参数 ---
        'input_dim': input_dim,
        'hidden_dims': hidden_dim,
        'n_layers': 1,
        'lambda_AE': 1.0,
        'lambda_outcome': 10.0,
        'seed': 2026,

        # --- 工具参数 ---
        'cuda': 1 if torch.cuda.is_available() else 0
    }

    # 2. 用用户传入的 kwargs 更新配置 (覆盖默认值)
    # 例如：传入 n_layers=2 会覆盖上面的 'n_layers': 1
    cfg.update(kwargs)

    # 3. 将字典转换为对象 (Namespace)
    # 这样 model.py 里的 args.input_dim 就能正常工作了，不需要改 model 代码
    args = SimpleNamespace(**cfg)

    # ================= 以下逻辑保持不变 =================

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # 路径处理
    if not args.input_path.endswith('/'):
        args.input_path += '/'

    path_train = args.input_path + args.filename_train
    path_valid = args.input_path + args.filename_valid
    path_data = args.input_path + args.filename_data
    path_test = args.input_path + args.filename_test

    # 加载数据
    data_train = AKIData(path_train)
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=False)
    data_valid = AKIData(path_valid)
    dataloader_valid = DataLoader(data_valid, batch_size=args.batch_size, shuffle=True)

    # 创建输出目录
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    writer = SummaryWriter(args.log_path)

    # 初始化模型 (Model 接收 args 对象)
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

    # 生成最终表示
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
