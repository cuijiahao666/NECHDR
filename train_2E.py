from math import exp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import fetch_dataloader_2E
from models.loss import HDRFlow_Loss_2E
from models.model_2E import HDRFlow
from models.IFRNet_s_comparable import MSANet
from utils.utils import *
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os.path as osp

def get_args():
    parser = argparse.ArgumentParser(description='HDRFlow',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_vimeo_dir", type=str, default='./data//vimeo_septuplet',
                        help='dataset directory'),
    parser.add_argument("--dataset_test_dir", type=str, default='./data/HDR_Synthetic_Test_Dataset-001',
                        help='dataset directory'),
    parser.add_argument('--logdir', type=str, default='./checkpoints/checkpoints_2E_ours',
                        help='target log directory')
    parser.add_argument('--num_workers', type=int, default=8, metavar='N',
                        help='number of workers to fetch data (default: 8)')
    parser.add_argument('--resume', type=str, default="",
                        help='load model from a .pth file')
    parser.add_argument('--seed', type=int, default=443, metavar='S',
                        help='random seed (default: 443)')
    parser.add_argument('--init_weights', action='store_true', default=False,
                        help='init model weights')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--lr_decay_epochs', type=str, 
                        default="100,100:2", help='the epochs to decay lr: the downscale rate')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='start epoch of training (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='training batch size (default: 16)')
    parser.add_argument('--val_batch_size', type=int, default=1, metavar='N',
                        help='testing batch size (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_results', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default="./output_results/2E_train")
    parser.add_argument('--nframes', type=int, default=3)
    parser.add_argument('--nexps', type=int, default=2)
    parser.add_argument('--tone_low', default=False, action='store_true')
    parser.add_argument('--tone_ref', default=True, action='store_true')
    return parser.parse_args()


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    for batch_idx, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)
        ldrs = [x.to(device) for x in batch_data['ldrs']]
        expos = [x.to(device) for x in batch_data['expos']]
        hdrs = [x.to(device) for x in batch_data['hdrs']]
        rev_ldrs = [x.to(device) for x in batch_data['rev_ldrs']]
        rev_expos = [x.to(device) for x in batch_data['rev_expos']]

        #--------------------add noise to low expos image-------------------
        prob = np.random.uniform()
        perturb_low_expo_imgs(args, ldrs, expos, prob)
        perturb_low_expo_imgs(args, rev_ldrs, rev_expos, prob)
        pred_hdr, loss_rec, loss_geo, loss_msa = model(ldrs, hdrs, rev_ldrs, expos)
        loss = loss_rec + 0.01 * loss_geo + 0.01 * loss_msa
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f} %)]\tLoss: {:.6f}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:3f})'.format(
                epoch,
                batch_idx,
                len(train_loader),
                100. * batch_idx / len(train_loader),
                loss.item(),
                batch_time=batch_time,
                data_time=data_time
            ))

def validation(args, model, device, val_loader, optimizer, epoch):
    model.eval()
    n_val = len(val_loader)
    val_psnr = AverageMeter()
    val_mu_psnr = AverageMeter()
    val_ssim = AverageMeter()
    val_mu_ssim = AverageMeter()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            ldrs = [x.to(device) for x in batch_data['ldrs']]
            expos = [x.to(device) for x in batch_data['expos']]
            gt_hdr = batch_data['hdr']

            padder = InputPadder(ldrs[0].shape, divis_by=16)
            pad_ldrs = padder.pad(ldrs)
            pred_hdr = model(pad_ldrs, 0, 0, expos, test_mode=True)
            pred_hdr = padder.unpad(pred_hdr)
            cur_ldr = ldrs[1]
            pred_hdr = torch.squeeze(pred_hdr.detach().cpu()).numpy().astype(np.float32).transpose(1,2,0)
            cur_ldr = torch.squeeze(ldrs[1].cpu()).numpy().astype(np.float32).transpose(1,2,0)
            gt_hdr = torch.squeeze(gt_hdr.cpu()).numpy().astype(np.float32).transpose(1,2,0)
            Y = 0.299 * cur_ldr[:, :, 0] + 0.587 * cur_ldr[:, :, 1] + 0.114 * cur_ldr[:, :, 2]
            Y = Y[:, :, None]
            if expos[1] <= 1.:
                mask = Y < 0.2
            else:
                mask = Y > 0.8
            cur_linear_ldr = ldr_to_hdr(ldrs[1], expos[1])
            cur_linear_ldr = torch.squeeze(cur_linear_ldr.cpu()).numpy().astype(np.float32).transpose(1,2,0)
            pred_hdr = (~mask) * cur_linear_ldr + (mask) * pred_hdr
            pred_hdr_tm = tonemap(pred_hdr)
            gt_hdr_tm = tonemap(gt_hdr)      

            psnrL = psnr(pred_hdr, gt_hdr)
            ssimL = ssim(gt_hdr, pred_hdr, multichannel=True, channel_axis=2, data_range=gt_hdr.max()-gt_hdr.min())
            psnrT = psnr(pred_hdr_tm, gt_hdr_tm)
            ssimT = ssim(gt_hdr_tm, pred_hdr_tm, multichannel=True, channel_axis=2, data_range=gt_hdr_tm.max()-gt_hdr_tm.min())

            val_psnr.update(psnrL.item())
            val_mu_psnr.update(psnrT.item())
            val_ssim.update(ssimL.item())
            val_mu_ssim.update(ssimT.item())
            # if args.save_results:
            #     hdr_output_dir = os.path.join(args.save_dir, 'hdr_output')
            #     if not osp.exists(hdr_output_dir):
            #         os.makedirs(hdr_output_dir)
            #     cv2.imwrite(os.path.join(hdr_output_dir, '{:0>3d}_pred.png'.format(batch_idx+1)), (pred_hdr_tm*255.)[:,:,[2,1,0]].astype('uint8'))
    print('Test set: Number: {}'.format(n_val))
    print('Test set: Average PSNR-l: {:.4f}, PSNR-mu: {:.4f}'.format(val_psnr.avg, val_mu_psnr.avg))
    save_dict = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(save_dict, os.path.join(args.logdir, 'checkpoint_%s.pth' % (epoch+1)))
    with open(os.path.join(args.logdir, 'checkpoint.json'), 'a') as f:
        f.write('epoch:' + str(epoch) + '\n')
        f.write('Validation set: Average PSNR-l: {:.4f}, PSNR-mu: {:.4f}, SSIM-l: {:.4f}, SSIM-mu: {:.4f}\n'.format(val_psnr.avg, val_mu_psnr.avg, val_ssim.avg, val_mu_ssim.avg))
    
    


def main():
    args = get_args()
    if args.seed is not None:
        set_random_seed(args.seed)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model
    model = MSANet()
    if args.init_weights:
        init_parameters(model)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    model.to(device)
    model = nn.DataParallel(model)

    if args.resume:
        if os.path.isfile(args.resume):
            print("===> Loading checkpoint from: {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("===> Loaded checkpoint: epoch {}".format(checkpoint['epoch']))
        else:
            print("===> No checkpoint is founded at {}.".format(args.resume))
    
    train_loader, val_loader = fetch_dataloader_2E(args)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch)
        train(args, model, device, train_loader, optimizer, epoch)
        validation(args, model, device, val_loader, optimizer, epoch)

if __name__ == '__main__':
    main()
