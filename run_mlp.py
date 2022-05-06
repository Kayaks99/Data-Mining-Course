import argparse
import glob
import os
import numpy as np
import pandas as pd
import torch.random
import torch.utils.data as data
from pandas import read_csv
import random
import time
from torch.optim import lr_scheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter
import datetime
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import StandardScaler


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class Dataset(data.Dataset):
    def __init__(self, input_data, label=None, split='train'):
        super(Dataset, self).__init__()
        self.data = input_data
        self.label = label
        self.split = split

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        input_data = torch.from_numpy(self.data[index])
        if self.split != 'test':
            target_label = torch.from_numpy(np.array([self.label[index]])).long()
            return input_data, target_label
        else:
            return input_data


class MLP(torch.nn.Module):
    def __init__(self, input_dim=73, hsize=256, cls=2):
        self.hsize = hsize
        super(MLP, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_dim, self.hsize, 1)
        self.conv2 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        # self.conv3 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv4 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv5 = torch.nn.Conv1d(self.hsize+input_dim, self.hsize, 1)
        self.conv6 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        # self.conv7 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv8 = torch.nn.Conv1d(self.hsize, cls, 1)

        # self.bn1 = torch.nn.BatchNorm1d(self.hsize)
        # self.bn2 = torch.nn.BatchNorm1d(self.hsize)
        # # self.bn3 = torch.nn.BatchNorm1d(self.hsize)
        # self.bn4 = torch.nn.BatchNorm1d(self.hsize)
        # self.bn5 = torch.nn.BatchNorm1d(self.hsize)
        # self.bn6 = torch.nn.BatchNorm1d(self.hsize)
        # # self.bn7 = torch.nn.BatchNorm1d(self.hsize)

        self.actv_fn = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        x1 = self.actv_fn(self.conv1(x))
        x2 = self.actv_fn(self.conv2(x1))
        # x3 = self.actv_fn(self.bn3(self.conv3(x2)))
        x3 = x2
        x4 = self.actv_fn(self.conv4(x3))
        x5 = self.actv_fn(self.conv5(torch.cat([x, x4], dim=1)))
        x6 = self.actv_fn(self.conv6(x5))
        # x7 = self.actv_fn(self.bn7(self.conv7(x6)))
        x7 = x6
        x8 = self.conv8(x7)
        return x8


def get_latest_epoch(resume_dir):
    os.makedirs(resume_dir, exist_ok=True)
    str_epoch = [file.split("_")[0] for file in os.listdir(resume_dir) if file.endswith("_states.pth")]
    int_epoch = [int(i) for i in str_epoch]
    return None if len(int_epoch) == 0 else str_epoch[int_epoch.index(max(int_epoch))]


def load_networks(model, epoch, device, resume_dir):
    load_filename = '{}_model.pth'.format(epoch)
    load_path = os.path.join(resume_dir, load_filename)
    if not os.path.isfile(load_path):
        print('cannot load', load_path)
        return None
    state_dict = torch.load(load_path, map_location=device)
    print('load path', load_path)
    model.load_state_dict(state_dict, strict=False)
    # print(state_dict)
    return model


def save_networks(model, save_dir, epoch, other_states=None, back_gpu=True):
    if other_states is None:
        other_states = {}
    save_filename = '{}_model.pth'.format(epoch)
    save_path = os.path.join(save_dir, save_filename)
    try:
        model.cpu()
        torch.save(model.state_dict(), save_path)
        if back_gpu:
            model.cuda()
    except Exception as e:
        print("save net:", e)
    save_filename = '{}_states.pth'.format(epoch)
    save_path = os.path.join(save_dir, save_filename)
    torch.save(other_states, save_path)


def test(model, dataloader):
    model.eval()
    predicted = []
    predicted_score = []
    labels = []
    with torch.no_grad():
        print('Evaluate.................')
        for i, batch in tqdm(enumerate(dataloader)):
            input_data, target_label = batch
            input_data = input_data.cuda()
            target_label = target_label.cuda()
            labels.append(target_label.squeeze(-1))
            pre_l = model(input_data.unsqueeze(-1)).squeeze(-1)  # [B, 2]
            pre_l = torch.softmax(pre_l, 1)
            predicted.append(torch.argmax(pre_l, dim=1))
            predicted_score.append(pre_l[..., 1])
    predicted = torch.cat(predicted).detach().cpu()
    predicted_score = torch.cat(predicted_score).detach().cpu()
    labels = torch.cat(labels).detach().cpu()
    f1 = f1_score(labels, predicted)
    auc = roc_auc_score(labels, predicted_score)
    acc = accuracy_score(labels, predicted)
    return acc, f1, auc


def generate_test_score(model, dataloader):
    print('generate_test_score...........')
    model.eval()
    predicted_score = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            input_data = batch
            input_data = input_data.cuda()
            pre_l = model(input_data.unsqueeze(-1)).squeeze(-1)  # [B, 2]
            pre_l = torch.softmax(pre_l, 1)
            predicted_score.append(pre_l[..., 1])

    predicted_score = torch.cat(predicted_score).detach().cpu().numpy().tolist()
    predicted_score = [round(d, 6) for d in predicted_score]
    id = [i for i in range(800000, 800000 + len(predicted_score))]

    df = pd.DataFrame(zip(id, predicted_score), columns=['id', 'isDefault'])
    df.to_csv('testA_submit.csv', index=False)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--print_freq',
        type=int,
        default=100,
        help='frequency of showing training results on console')
    parser.add_argument('--lr',
                        type=float,
                        default=0.005,
                        help='initial learning rate')
    parser.add_argument('--full_train',
                        type=int,
                        default=0,
                        help='train on full dataset(no val dataset)')
    parser.add_argument('--gpu_ids',
                        type=str,
                        default='0')
    parser.add_argument('--checkpoints_dir',
                        default='./checkpoints',
                        type=str)
    parser.add_argument('--lr_decay_iters',
                        default=10000000,
                        type=int)
    # parser.add_argument('--lr_decay_iters',
    #                     default=100000,
    #                     type=int)
    parser.add_argument('--lr_decay_exp',
                        default=0.1,
                        type=int)
    parser.add_argument('--epoch',
                        default=1000,
                        type=int)
    parser.add_argument('--lr_policy',
                        default="iter_exponential_decay",
                        type=str)
    parser.add_argument('--save_epoch_freq',
                        default=20,
                        type=int)
    opt = parser.parse_args()
    opt.gpu_ids = [
        int(x) for x in opt.gpu_ids.split(',') if x.strip() and int(x) >= 0
    ]
    cur_device = torch.device('cuda:{}'.format(opt.gpu_ids[0]) if opt.
                              gpu_ids else torch.device('cpu'))

    data_path = 'processed_data/data_for_model2.csv'
    label_path = 'processed_data/label_for_model.csv'
    all_data = read_csv(data_path)
    all_data = all_data.groupby('sample')

    train_data = all_data.get_group('train').drop(['sample'], axis=1)
    test_data = all_data.get_group('test').drop(['sample'], axis=1)
    train_data = np.array(train_data).astype(np.float32)  # (612742, 77)
    test_data = np.array(test_data).astype(np.float32)  # (200000, 77)

    train_len, test_len = train_data.shape[0], test_data.shape[0]
    train_test_data = np.concatenate([train_data, test_data], axis=0)

    valid_idx = np.max(train_data, axis=0) != np.min(train_data, axis=0)
    train_test_data = train_test_data[:, valid_idx]
    valid_idx = np.max(train_test_data, axis=0) != np.inf
    train_test_data = train_test_data[:, valid_idx]
    train_test_data = train_test_data / np.max(train_test_data, axis=0) * 2 - 1
    input_dim = train_test_data.shape[1]
    train_data = train_test_data[:train_len]
    test_data = train_test_data[train_len:]
    assert test_data.shape[0] == test_len

    label = read_csv(label_path)
    label = np.array(label).astype(np.float32)[..., 0]  # [612742, 1]

    idx = list(np.arange(0, train_data.shape[0]))
    np.random.shuffle(idx)

    train_idx = idx[:int(train_data.shape[0] * 0.8)]
    val_idx = idx[int(train_data.shape[0] * 0.8):]

    all_val_data = train_data[val_idx]
    all_train_data = train_data[train_idx]

    all_train_label = label[train_idx]
    all_val_label = label[val_idx]

    tb_writer = SummaryWriter(
        os.path.join(
            opt.checkpoints_dir,
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    with torch.no_grad():
        print(opt.checkpoints_dir + "/*_model.pth")
        if len([n for n in glob.glob(opt.checkpoints_dir + "/*_model.pth") if
                os.path.isfile(n)]) > 0:  # resume training
            resume_dir = opt.checkpoints_dir
            resume_iter = get_latest_epoch(resume_dir)
            if resume_iter is None:
                epoch_count = 1
                total_steps = 0
                print("No previous checkpoints, start from scratch!!!!")
            else:
                states = torch.load(
                    os.path.join(resume_dir, '{}_states.pth'.format(resume_iter)), map_location=cur_device)
                print(os.path.join(resume_dir, '{}_states.pth'.format(resume_iter)))
                print(states)
                epoch_count = states['epoch_count']
                total_steps = states['total_steps']
                print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                print('Continue training from {} epoch'.format(epoch_count))
                print(f"Iter: {total_steps}")
                print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                del states
            model = MLP(input_dim=input_dim).to(cur_device)
            model = load_networks(model, epoch_count, cur_device, resume_dir)
        else:  # Train from scratch
            model = MLP(input_dim=input_dim).to(cur_device)
            epoch_count = 1
            total_steps = 0

    optimizer = torch.optim.SGD(params=model.parameters(), lr=opt.lr, momentum=0.9)

    def lambda_rule(it):
        lr_l = pow(opt.lr_decay_exp, it / opt.lr_decay_iters)
        return lr_l

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    model.train()

    def worker_init_fn(worker_id):
        np.random.seed(
            (worker_id + torch.initial_seed() + np.floor(time.time()).astype(np.int64)) % np.iinfo(np.int32).max)
    if opt.full_train > 0:
        all_train_data = np.concatenate([all_train_data, all_val_data], 0)
        all_train_label = np.concatenate([all_train_label, all_val_label], 0)
    train_dataset = Dataset(all_train_data, all_train_label, 'train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    val_dataset = Dataset(all_val_data, all_val_label, 'val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)
    test_dataset = Dataset(input_data=test_data, split='test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)
    train_dataset_size = len(train_dataloader)
    val_dataset_size = len(val_dataloader)
    print('# training data = {}'.format(train_dataset_size))
    print('# val data = {}'.format(val_dataset_size))

    if total_steps > 0:
        for i in range(total_steps):
            scheduler.step()

    criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.2, 0.8])).float().cuda())

    best_epoch = 0
    best_auc = 0
    for epoch in tqdm(range(epoch_count, opt.epoch)):
        if epoch_count >= opt.epoch - 1:
            break
        for i, batch in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()
            total_steps += 1

            input_data, target_label = batch
            input_data = input_data.cuda()
            target_label = target_label.cuda()
            pre_l = model(input_data.unsqueeze(-1)).squeeze(-1)

            loss = criterion(pre_l, target_label.squeeze(-1))
            if total_steps is not None and total_steps % opt.print_freq == 0:
                print('total_loss : ', loss.item())
                tb_writer.add_scalar('loss', float(loss.item()), total_steps)

            loss.backward()
            optimizer.step()

            if opt.lr_policy.startswith("iter"):
                scheduler.step()
                lr = optimizer.param_groups[0]['lr']
                if total_steps % opt.print_freq == 0:
                    print('learning rate = {:.7f}'.format(lr))
        try:
            if (epoch > 0 and epoch % opt.save_epoch_freq == 0) or epoch == opt.epoch - 1:
                other_states = {
                    'epoch_count': epoch,
                    'total_steps': total_steps,
                }
                print('saving model (epoch {}, total_steps {})'.format(epoch, total_steps))
                save_networks(model, opt.checkpoints_dir, epoch, other_states)
        except Exception as e:
            print(e)
        acc, f1, auc = test(model, val_dataloader)
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
        print('Epoch:{:4d} Acc:{:.4f}  F1-Score:{:.4f} AUC:{:.4f}'.format(epoch, acc, f1, auc))
        print('Best AUC:{:.4f}  Epoch:{:4d}'.format(best_auc, best_epoch))
        model.train()

    other_states = {
        'epoch_count': epoch,
        'total_steps': total_steps,
    }
    save_networks(model, opt.checkpoints_dir, epoch, other_states)

    generate_test_score(model, test_dataloader)


if __name__ == '__main__':
    main()
