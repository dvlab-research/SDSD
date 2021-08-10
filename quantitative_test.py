import cv2

import os.path as osp
import logging
import time
import argparse

import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)


def main():
    model = create_model(opt)

    print('mkdir finish')

    logger = logging.getLogger('base')

    for phase, dataset_opt in opt['datasets'].items():
        val_set = create_dataset(dataset_opt)
        val_loader = create_dataloader(val_set, dataset_opt, opt, None)

        pbar = util.ProgressBar(len(val_loader))
        psnr_rlt = {}
        psnr_rlt_avg = {}
        psnr_total_avg = 0.

        ssim_rlt = {}
        ssim_rlt_avg = {}
        ssim_total_avg = 0.

        for val_data in val_loader:
            folder = val_data['folder'][0]

            idx_d = val_data['idx']
            if psnr_rlt.get(folder, None) is None:
                psnr_rlt[folder] = []

            if ssim_rlt.get(folder, None) is None:
                ssim_rlt[folder] = []
            model.feed_data(val_data)

            model.test()

            visuals = model.get_current_visuals()
            rlt_img = util.tensor2img(visuals['rlt'])
            gt_img = util.tensor2img(visuals['GT'])

            psnr = util.calculate_psnr(rlt_img, gt_img)
            psnr_rlt[folder].append(psnr)

            ssim = util.calculate_ssim(rlt_img, gt_img)
            ssim_rlt[folder].append(ssim)

            pbar.update('Test {} - {}'.format(folder, idx_d))
        for k, v in psnr_rlt.items():
            psnr_rlt_avg[k] = sum(v) / len(v)
            psnr_total_avg += psnr_rlt_avg[k]

        for k, v in ssim_rlt.items():
            ssim_rlt_avg[k] = sum(v) / len(v)
            ssim_total_avg += ssim_rlt_avg[k]

        psnr_total_avg /= len(psnr_rlt)
        ssim_total_avg /= len(ssim_rlt)
        print(psnr_total_avg,ssim_total_avg)
        log_s = '# Validation # PSNR: {:.4e}:'.format(psnr_total_avg)
        for k, v in psnr_rlt_avg.items():
            log_s += ' {}: {:.4e}'.format(k, v)
        logger.info(log_s)
        print(log_s)

        log_s = '# Validation # SSIM: {:.4e}:'.format(ssim_total_avg)
        for k, v in ssim_rlt_avg.items():
            log_s += ' {}: {:.4e}'.format(k, v)
        logger.info(log_s)
        print(log_s)

if __name__ == '__main__':
    main()
