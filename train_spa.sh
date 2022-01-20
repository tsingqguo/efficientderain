python ./train.py \
--baseroot "/DATASET/DERAIN/SPA_Training" \
--load_name "" \
--multi_gpu "false"  \
--save_path "./models/models_SPA" \
--sample_path "./samples" \
--save_mode 'epoch' \
--save_by_epoch 1 \
--save_by_iter 10000 \
--lr_g 0.0002 \
--b1 0.5 \
--b2 0.999 \
--weight_decay 0.0 \
--train_batch_size 16 \
--epochs 20 \
--lr_decrease_epoch 7 \
--num_workers 1 \
--crop_size 256 \
--no_gpu "false" \
--rainaug "false" \


