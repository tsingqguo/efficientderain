import time
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
#import encoding
from torchvision import transforms

import pytorch_ssim
import dataset
import utils

def Pre_train(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    #torch.cuda.set_device(1)
    
    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    save_folder = opt.save_path
    sample_folder = opt.sample_path
    utils.check_path(save_folder)
    utils.check_path(sample_folder)

    # Loss functions
    if opt.no_gpu == False:
        criterion_L1 = torch.nn.L1Loss().cuda()
        criterion_L2 = torch.nn.MSELoss().cuda()
        #criterion_rainypred = torch.nn.L1Loss().cuda()
        criterion_ssim = pytorch_ssim.SSIM().cuda()
    else: 
        criterion_L1 = torch.nn.L1Loss()
        criterion_L2 = torch.nn.MSELoss()
        #criterion_rainypred = torch.nn.L1Loss().cuda()
        criterion_ssim = pytorch_ssim.SSIM()

    # Initialize Generator
    generator = utils.create_generator(opt)

    # To device
    if opt.no_gpu == False:
        if opt.multi_gpu:
            generator = nn.DataParallel(generator)
            generator = generator.cuda()
        else:
            generator = generator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    #optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    
    # pretrained model
    #encnet = encoding.models.get_model('Encnet_ResNet50s_PContext', pretrained=True).cuda()
    #encnet.eval()
    #resnet = (torch.nn.Sequential(*list(encnet.children())[:1]))[0]
    #resnet.eval()
    #encnet_feat = torch.nn.Sequential(*list(resnet.children())[:1])
    #encnet_feat.eval()

    #for param in encnet.parameters():
    #    param.requires_grad = False
    print("pretrained models loaded")

    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, optimizer):
        target_epoch = opt.epochs - opt.lr_decrease_epoch
        remain_epoch = opt.epochs - epoch
        if epoch >= opt.lr_decrease_epoch:
            lr = opt.lr_g * remain_epoch / target_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        # Define the name of trained model
        """
        if opt.save_mode == 'epoch':
            model_name = 'KPN_single_image_epoch%d_bs%d_mu%d_sigma%d.pth' % (epoch, opt.train_batch_size, opt.mu, opt.sigma)
        if opt.save_mode == 'iter':
            model_name = 'KPN_single_image_iter%d_bs%d_mu%d_sigma%d.pth' % (iteration, opt.train_batch_size, opt.mu, opt.sigma)
        """
        if opt.save_mode == 'epoch':
            model_name = 'KPN_rainy_image_epoch%d_bs%d.pth' % (epoch, opt.train_batch_size)
        if opt.save_mode == 'iter':
            model_name = 'KPN_rainy_image_iter%d_bs%d.pth' % (iteration, opt.train_batch_size)
        save_model_path = os.path.join(opt.save_path, model_name)
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Handle multiple GPUs
    #os.environ["CUDA_VISIBLE_DEVICES"] = ""    
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    #if opt.no_gpu == False:
        #opt.train_batch_size *= gpu_num
        #opt.val_batch_size *= gpu_num
        #opt.num_workers *= gpu_num
   
    #print(opt.multi_gpu)
    '''
    print(opt.no_gpu == False)
    print(opt.no_gpu)
    print(gpu_num)
    print(opt.train_batch_size)
    '''
    
    # Define the dataset
    trainset = dataset.DenoisingDataset(opt)
    print('The overall number of training images:', len(trainset))

    # Define the dataloader
    train_loader = DataLoader(trainset, batch_size = opt.train_batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    
    # For loop training
    for epoch in range(opt.epochs):
        for i, (true_input, true_target) in enumerate(train_loader):

            #print("in epoch %d" % i)

            if opt.no_gpu == False:
                # To device
                true_input = true_input.cuda()
                true_target = true_target.cuda()

            # Train Generator
            optimizer_G.zero_grad()
            fake_target = generator(true_input, true_input)
            
            
            ssim_loss = -criterion_ssim(true_target, fake_target)

            '''
            #trans for enc_net
            enc_trans = transforms.Compose([transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
            fake_target_norm = torch.from_numpy(np.zeros(fake_target.size())).cuda()
            true_target_norm = torch.from_numpy(np.zeros(true_target.size())).cuda()
            for j in range(fake_target.size()[0]):
                fake_target_norm[j] = enc_trans(fake_target[j])
                true_target_norm[j] = enc_trans(true_target[j])
            ''' 

            #print(fake_target_norm.size())
            #enc_pred = encnet.evaluate(fake_target_norm.type(torch.FloatTensor).cuda())
            #enc_pred = encnet(fake_target_norm.type(torch.FloatTensor).cuda())[0]
            #enc_gt = encnet(true_target_norm.type(torch.FloatTensor).cuda())[0]

            '''
            enc_feat_pred = encnet_feat(fake_target_norm.type(torch.FloatTensor).cuda())[0]
            enc_feat_gt = encnet_feat(true_target_norm.type(torch.FloatTensor).cuda())[0]
            '''

            #rain_layer_gt = true_input - true_target 
            #rain_layer_pred = true_input - fake_target
            #rainy_pred = true_input - (fake_target * rain_layer_pred) 
            #print(type(true_input))
            #print(type(fake_target))

            # L1 Loss
            Pixellevel_L1_Loss = criterion_L1(fake_target, true_target)
            #enc_loss = criterion_L1(enc_pred, enc_gt)
            #enc_feat_loss = criterion_L1(enc_feat_pred, enc_feat_gt)
            #Pixellevel_L2_Loss = criterion_L2(fake_target, true_target)
            #Pixellevel_L2_Loss = criterion_L2(rain_layer_pred, rain_layer_gt)
            #Loss_rainypred = criterion_rainypred(rainy_pred, true_input)
            
            # Overall Loss and optimize
            loss = Pixellevel_L1_Loss + 0.2*ssim_loss
            #loss = Pixellevel_L1_Loss
            #loss = Pixellevel_L1_Loss + Pixellevel_L2_Loss + Loss_rainypred
            loss.backward()
            optimizer_G.step()

            #check
            '''
            for j in encnet.named_parameters():
                print(j)
                break
            '''

            # Determine approximate time left
            iters_done = epoch * len(train_loader) + i
            iters_left = opt.epochs * len(train_loader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Loss: %.4f %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(train_loader), Pixellevel_L1_Loss.item(), ssim_loss.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(train_loader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), optimizer_G)

        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [true_input, fake_target, true_target]
            name_list = ['in', 'pred', 'gt']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'train_epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)
        '''
        ### Validation
        val_PSNR = 0
        num_of_val_image = 0

        for j, (true_input, true_target) in enumerate(val_loader):
            
            # To device
            # A is for input image, B is for target image
            true_input = true_input.cuda()
            true_target = true_target.cuda()

            # Forward propagation
            with torch.no_grad():
                fake_target = generator(true_input)

            # Accumulate num of image and val_PSNR
            num_of_val_image += true_input.shape[0]
            val_PSNR += utils.psnr(fake_target, true_target, 1) * true_input.shape[0]
        val_PSNR = val_PSNR / num_of_val_image

        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [true_input, fake_target, true_target]
            name_list = ['in', 'pred', 'gt']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'val_epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

        # Record average PSNR
        print('PSNR at epoch %d: %.4f' % ((epoch + 1), val_PSNR))
        '''
