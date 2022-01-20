python ./train.py \
--baseroot "./datasets/rain100H/train" \
--load_name "" \
--multi_gpu "true"  \
--save_path "./models/models_rain100H" \
--sample_path "./samples" \
--save_mode 'epoch' \
--save_by_epoch 250 \
--save_by_iter 10000 \
--lr_g 0.0002 \
--b1 0.5 \
--b2 0.999 \
--weight_decay 0.0 \
--train_batch_size 16 \
--epochs 5000 \
--lr_decrease_epoch 2000 \
--num_workers 1 \
--crop_size 256 \
--no_gpu "false" \
--rainaug "false" \


