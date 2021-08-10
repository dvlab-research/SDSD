import cv2

import os.path as osp
import logging
import time
import argparse

import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model
import numpy as np
import os

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)


def main():
    save_imgs = True
    model = create_model(opt)
    save_folder = './results/{}'.format(opt['name'])
    output_folder = osp.join(save_folder, 'images/output')
    input_folder = osp.join(save_folder, 'images/input')
    util.mkdirs(save_folder)
    util.mkdirs(output_folder)
    util.mkdirs(input_folder)

    print('mkdir finish')

    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)

    for phase, dataset_opt in opt['datasets'].items():
        val_set = create_dataset(dataset_opt)
        val_loader = create_dataloader(val_set, dataset_opt, opt, None)

        pbar = util.ProgressBar(len(val_loader))

        for val_data in val_loader:
            folder = val_data['folder'][0]

            idx_d = val_data['idx']
            model.feed_data(val_data)

            model.test()

            visuals = model.get_current_visuals()
            rlt_img = util.tensor2img(visuals['rlt'])

            mid_ix = dataset_opt['N_frames'] // 2
            input_img = util.tensor2img(visuals['LQ'][mid_ix])

            if save_imgs:
                try:
                    tag = '{}.{}'.format(val_data['folder'], idx_d[0].replace('/', '-'))
                    print(osp.join(output_folder, '{}.png'.format(tag)))
                    cv2.imwrite(osp.join(output_folder, '{}.png'.format(tag)), rlt_img)
                    cv2.imwrite(osp.join(input_folder, '{}.png'.format(tag)), input_img)

                except Exception as e:
                    print(e)
                    import ipdb; ipdb.set_trace()

            pbar.update('Test {} - {}'.format(folder, idx_d))



if __name__ == '__main__':
    main()
