import cv2
import numpy as np
import matplotlib.pyplot as plt 
from skimage.measure import compare_ssim, compare_psnr

SRC_PATH = "./results_k9"
DST_PATH = "./rainy_image_dataset/testing_results_k9"
MAXID = 1400

dst_path_gt = DST_PATH + "/ground_truth"
dst_path_in = DST_PATH + "/rainy_image"

for i in range(1, MAXID+1):
    print("dealing with %d" % i)
    index1 = ((i-1) // 14) + 1
    index2 = ((i-1) % 14) + 1
    file_gt = SRC_PATH + '/' + str(i) + '_gt.png'
    file_pred = SRC_PATH + '/' + str(i) + '_pred.png'
    file_gt_dst = dst_path_gt + '/' + str(index1) + '_' + str(index2) + '.jpg'
    file_in_dst = dst_path_in + '/' + str(index1) + '_' + str(index2) + '.jpg'

    img_gt = cv2.imread(file_gt)
    img_pred = cv2.imread(file_pred)
    cv2.imwrite(file_gt_dst, img_gt)
    cv2.imwrite(file_in_dst, img_pred)    
                                            
