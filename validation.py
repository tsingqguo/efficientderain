import argparse
import os
import torch
import numpy as np
import cv2
from skimage.metrics import structural_similarity

import utils
import dataset

def str2bool(v):
    #print(v)
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    #GPU parameters
    parser.add_argument('--no_gpu', default = False, help = 'True for CPU')
    # Saving, and loading parameters
    parser.add_argument('--save_name', type = str, default = './results_tmp', help = 'save the generated with certain epoch')
    parser.add_argument('--load_name', type = str, default = './models_k9_loss1/KPN_rainy_image_epoch170_bs16.pth', help = 'load the pre-trained model with certain epoch')
    #parser.add_argument('--load_name', type = str, default = './models/KPN_single_image_epoch120_bs16_mu0_sigma30.pth', help = 'load the pre-trained model with certain epoch')
    parser.add_argument('--test_batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--num_workers', type = int, default = 1, help = 'number of workers')
    # Initialization parameters
    parser.add_argument('--color', type = str2bool, default = True, help = 'input type')
    parser.add_argument('--burst_length', type = int, default = 1, help = 'number of photos used in burst setting')
    parser.add_argument('--blind_est', type = str2bool, default = True, help = 'variance map')
    parser.add_argument('--kernel_size', type = list, default = [3], help = 'kernel size')
    parser.add_argument('--sep_conv', type = str2bool, default = False, help = 'simple output type')
    parser.add_argument('--channel_att', type = str2bool, default = False, help = 'channel wise attention')
    parser.add_argument('--spatial_att', type = str2bool, default = False, help = 'spatial wise attention')
    parser.add_argument('--upMode', type = str, default = 'bilinear', help = 'upMode')
    parser.add_argument('--core_bias', type = str2bool, default = False, help = 'core_bias')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'initialization type of generator')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of generator')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, default = 'rainy_image_dataset/testing', help = 'images baseroot')
    parser.add_argument('--crop', type = str2bool, default = False, help = 'whether to crop input images')
    parser.add_argument('--crop_size', type = int, default = 512, help = 'single patch size')
    parser.add_argument('--geometry_aug', type = str2bool, default = False, help = 'geometry augmentation (scaling)')
    parser.add_argument('--angle_aug', type = str2bool, default = False, help = 'geometry augmentation (rotation, flipping)')
    parser.add_argument('--scale_min', type = float, default = 1, help = 'min scaling factor')
    parser.add_argument('--scale_max', type = float, default = 1, help = 'max scaling factor')
    parser.add_argument('--add_noise', type = str2bool, default = False, help = 'whether to add noise to input images')
    parser.add_argument('--mu', type = int, default = 0, help = 'Gaussian noise mean')
    parser.add_argument('--sigma', type = int, default = 30, help = 'Gaussian noise variance: 30 | 50 | 70')
    opt = parser.parse_args()
    print(opt)
    
#    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # ----------------------------------------
    #                   Test
    # ----------------------------------------
    # Initialize
    if opt.no_gpu:
        generator = utils.create_generator(opt)
    else:
        generator = utils.create_generator(opt).cuda()

    '''
    parm={}
    for name,parameters in generator.named_parameters():
        print(name,':',parameters.size())
        parm[name]=parameters.detach().cpu().numpy()
    print(parm['conv_final.weight'])
    print(parm['conv_final.bias'])
    '''

    test_dataset = dataset.DenoisingValDataset(opt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = opt.test_batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    sample_folder = opt.save_name
    utils.check_path(sample_folder)

    psnr_sum, psnr_ave, ssim_sum, ssim_ave, eval_cnt = 0, 0, 0, 0, 0
    
    # forward
    for i, (true_input, true_target, height_origin, width_origin) in enumerate(test_loader):

        # To device
        if opt.no_gpu:
            true_input = true_input
            true_target = true_target
        else:
            true_input = true_input.cuda()
            true_target = true_target.cuda()            

        # Forward propagation
        with torch.no_grad():
            #print(true_input.size()) 
            fake_target = generator(true_input, true_input)
        
        #print(fake_target.shape, true_input.shape)

        # Save
        print('The %d-th iteration' % (i))
        img_list = [true_input, fake_target, true_target]
        name_list = ['in', 'pred', 'gt']
        sample_name = '%d' % (i+1)
        utils.save_sample_png(sample_folder = sample_folder, sample_name = '%d' % (i + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255, height = height_origin, width = width_origin)
        
        # Evaluation
        #psnr_sum = psnr_sum + utils.psnr(cv2.imread(sample_folder + '/' + sample_name + '_' + name_list[1] + '.png').astype(np.float32), cv2.imread(sample_folder + '/' + sample_name + '_' + name_list[2] + '.png').astype(np.float32))
        img_pred_recover = utils.recover_process(fake_target, height = height_origin, width = width_origin)
        img_gt_recover = utils.recover_process(true_target, height = height_origin, width = width_origin)
        #psnr_sum = psnr_sum + utils.psnr(utils.recover_process(fake_target, height = height_origin, width = width_origin), utils.recover_process(true_target, height = height_origin, width = width_origin))
        psnr_sum = psnr_sum + utils.psnr(img_pred_recover, img_gt_recover)
        ssim_sum = ssim_sum + compare_ssim(img_gt_recover, img_pred_recover, multichannel = True, data_range = 255) 
        eval_cnt = eval_cnt + 1
        
    psnr_ave = psnr_sum / eval_cnt
    ssim_ave = ssim_sum / eval_cnt
    psnr_file = "./data/psnr_data.txt"
    ssim_file = "./data/ssim_data.txt"
    psnr_content = opt.load_name + ": " + str(psnr_ave) + "\n"
    ssim_content = opt.load_name + ": " + str(ssim_ave) + "\n"
    utils.text_save(content = psnr_content, filename = psnr_file)
    utils.text_save(content = ssim_content, filename = ssim_file)
    
    
