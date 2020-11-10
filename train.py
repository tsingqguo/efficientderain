import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse

import trainer

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
    # Pre-train, saving, and loading parameters
    parser.add_argument('--save_path', type = str, default = './models_k9_loss14_ft', help = 'saving path that is a folder')
    parser.add_argument('--sample_path', type = str, default = './samples', help = 'training samples path that is a folder')
    parser.add_argument('--save_mode', type = str, default = 'epoch', help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type = int, default = 10, help = 'interval between model checkpoints (by epochs)')
    parser.add_argument('--save_by_iter', type = int, default = 100000, help = 'interval between model checkpoints (by iterations)')
    parser.add_argument('--load_name', type = str, default = '', help = 'load the pre-trained model with certain epoch')
    # GPU parameters
    parser.add_argument('--no_gpu', type = str2bool, default = False, help = 'True for CPU')
    parser.add_argument('--multi_gpu', type = str2bool, default = False, help = 'True for more than 1 GPU')
    #parser.add_argument('--multi_gpu', type = bool, default = False, help = 'True for more than 1 GPU')
    parser.add_argument('--gpu_ids', type = str, default = '0, 1, 2, 3', help = 'gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = str2bool, default = True, help = 'True for unchanged input data type')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 100, help = 'number of epochs of training')
    parser.add_argument('--train_batch_size', type = int, default = 16, help = 'size of the batches')
    parser.add_argument('--lr_g', type = float, default = 0.0002, help = 'Adam: learning rate for G / D')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay for optimizer')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 20, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
    # Initialization parameters
    parser.add_argument('--color', type = str2bool, default = True, help = 'input type')
    parser.add_argument('--burst_length', type = int, default = 1, help = 'number of photos used in burst setting')
    parser.add_argument('--blind_est', type = str2bool, default = True, help = 'variance map')
    parser.add_argument('--kernel_size', type = str2bool, default = [3], help = 'kernel size')
    parser.add_argument('--sep_conv', type = str2bool, default = False, help = 'simple output type')
    parser.add_argument('--channel_att', type = str2bool, default = False, help = 'channel wise attention')
    parser.add_argument('--spatial_att', type = str2bool, default = False, help = 'spatial wise attention')
    parser.add_argument('--upMode', type = str, default = 'bilinear', help = 'upMode')
    parser.add_argument('--core_bias', type = str2bool, default = False, help = 'core_bias')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'initialization type of generator')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of generator')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, default = './rainy_image_dataset/training', help = 'images baseroot')
    parser.add_argument('--rainaug', type = str2bool, default = False, help = 'true for using rainaug')
    parser.add_argument('--crop_size', type = int, default = 256, help = 'single patch size')
    parser.add_argument('--geometry_aug', type = str2bool, default = False, help = 'geometry augmentation (scaling)')
    parser.add_argument('--angle_aug', type = str2bool, default = False, help = 'geometry augmentation (rotation, flipping)')
    parser.add_argument('--scale_min', type = float, default = 1, help = 'min scaling factor')
    parser.add_argument('--scale_max', type = float, default = 1, help = 'max scaling factor')
    parser.add_argument('--mu', type = int, default = 0, help = 'Gaussian noise mean')
    parser.add_argument('--sigma', type = int, default = 30, help = 'Gaussian noise variance: 30 | 50 | 70')
    opt = parser.parse_args()
    print(opt)
    
    
    '''
    print(opt.no_gpu)
    print(opt.no_gpu==False)
    '''
    
    ''' 
    # ----------------------------------------
    #        Choose CUDA visible devices
    # ----------------------------------------
    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        print('Multi-GPU mode, %s GPUs are used' % (opt.gpu_ids))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('Single-GPU mode')
    
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    '''


    # ----------------------------------------
    #       Choose pre / continue train
    # ----------------------------------------
    trainer.Pre_train(opt)
    
