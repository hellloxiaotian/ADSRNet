import os
import math
from decimal import Decimal

import utility

import torch
from torch import tensor
import torch.nn.utils as utils
from tqdm import tqdm
import numpy as np
import cv2

from thop import profile
import time
from ptflops import get_model_complexity_info

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()  # learning rate

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {}'.format(epoch, lr)
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(self.scale)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

        learning_rate = self.decay_learning_rate(epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate

        self.optimizer.schedule()

    def decay_learning_rate(self, epoch):
        lr = self.args.lr * (0.5 ** (epoch // 300))
        return lr

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        max_psnr=0
        min_psnr=100000
        sum_psnr=0

        max_ssim=0
        min_ssim=100000
        sum_ssim=0

        max_nmi=0
        min_nmi=100000
        sum_nmi=0

        max_hd=0
        min_hd=100000
        sum_hd=0

        max_nrmse=0
        min_nrmse=100000
        sum_nrmse=0

        max_lpips=0
        min_lpips=100000
        sum_lpips=0

        max_brisque=0
        min_brisque=100000
        sum_brisque=0

        max_fsim=0
        min_fsim=100000
        sum_fsim=0

        max_hpsi=0
        min_hpsi=100000
        sum_hpsi=0

        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)

                    sr = utility.quantize(sr, self.args.rgb_range)
                    max_psnr= max(max_psnr,peak_signal_noise_ratio(hr, sr))
                    min_psnr= min(min_psnr,peak_signal_noise_ratio(hr, sr))
                    sum_psnr+=peak_signal_noise_ratio(hr, sr)
                    (score1,diff1)=ssim(hr, sr,win_size=3, full=True)
                    i1=np_to_torch(sr)
                    i2=np_to_torch(hr)

                    nmi_score_value = nmi(hr, sr)   #nmi
                    nrmse_score_value = nrmse(hr, sr)  #nrmse
                    hd_score_value = hd(hr, sr)  #hd

                    max_ssim= max(max_ssim,score1)
                    min_ssim= min(min_ssim,score1)
                    sum_ssim+=score1
                    avg_psnr=sum_psnr/(j+1)


                    avg_ssim=sum_ssim/(j+1)

                    max_nmi= max(max_nmi,nmi_score_value)
                    min_nmi= min(min_nmi,nmi_score_value)
                    sum_nmi+=nmi_score_value
                    avg_nmi=sum_nmi/(j+1)

                    max_nrmse= max(max_nrmse,nrmse_score_value)
                    min_nrmse= min(min_nrmse,nrmse_score_value)
                    sum_nrmse+=nrmse_score_value
                    avg_nrmse=sum_nrmse/(j+1)

                    max_hd= max(max_hd,hd_score_value)
                    min_hd= min(min_hd,hd_score_value)
                    sum_hd+=hd_score_value
                    avg_hd=sum_hd/(j+1)

                    score = loss_fn_alex(i1, i2).item()
                    max_lpips= max(max_lpips,score)
                    min_lpips= min(min_lpips,score)
                    sum_lpips+=score
                    avg_lpips=sum_lpips/(j+1)

                    score_b=piq.brisque(i1)
                    score_h=piq.haarpsi(i1,i2)
                    score_f=piq.fsim(i1,i2)

                    max_hpsi= max(max_hpsi,score_h)
                    min_hpsi= min(min_hpsi,score_h)
                    sum_hpsi+=score_h
                    avg_hpsi=sum_hpsi/(j+1)

                    max_brisque= max(max_brisque,score_b)
                    min_brisque= min(min_brisque,score_b)
                    sum_brisque+=score_b
                    avg_brisque=sum_brisque/(j+1)

                    max_fsim= max(max_fsim,score_f)
                    min_fsim= min(min_fsim,score_f)
                    sum_fsim+=score_f
                    avg_fsim=sum_fsim/(j+1)

                    plot_image_grid([hr,sr],factor=4, nrow=2)
                    print("Shape of the output image: ", sr.shape)
                    print('Till Iteration %d\nPSNR: Max: %.3f Min: %.3f Avg: %.3f\nSSIM: Max: %.3f Min: %.3f Avg: %.3f\n' % (j+1,max_psnr,min_psnr,avg_psnr,max_ssim,min_ssim,avg_ssim))
                    print('NMI: Max: %.3f Min: %.3f Avg: %.3f\nNRMSE: Max: %.3f Min: %.3f Avg: %.3f\nHausdorff Distance: Max: %.3f Min: %.3f Avg: %.3f\n' % (max_nmi,min_nmi,avg_nmi,max_nrmse,min_nrmse,avg_nrmse,max_hd,min_hd,avg_hd))
                    print('LPIPS: Max: %.3f Min: %.3f Avg: %.3f\nBRISQUE: Max: %.3f Min: %.3f Avg: %.3f\nFSIM: Max: %.3f Min: %.3f Avg: %.3f\nHPSI: Max: %.3f Min: %.3f Avg: %.3f\n' % (max_lpips,min_lpips,avg_lpips,max_brisque,min_brisque,avg_brisque,max_fsim,min_fsim,avg_fsim,max_hpsi,min_hpsi,avg_hpsi))

                    save_list = [sr]
                    # self.ckp.log[-1, idx_data, idx_scale] += utility.calc_ssim(  # calc_ssim
                    #     sr, hr, scale, self.args.rgb_range, dataset=d
                    # )
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )

                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.5f} (Best: {:.5f} @epoch {})'.format(
                        # '[{} x{}]\tSSIM: {:.5f} (Best: {:.5f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1,
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.dend_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda:0')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch()
            return epoch >= self.args.epochs
