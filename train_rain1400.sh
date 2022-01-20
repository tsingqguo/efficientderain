python ./train.py \
--baseroot "./datasets/rain1400/train" \
--load_name "" \
--multi_gpu "false"  \
--save_path "./models/models_rain1400" \
--sample_path "./samples" \
--save_mode 'epoch' \
--save_by_epoch 10 \
--save_by_iter 10000 \
--lr_g 0.0002 \
--b1 0.5 \
--b2 0.999 \
--weight_decay 0.0 \
--train_batch_size 16 \
--epochs 250 \
--lr_decrease_epoch 50 \
--num_workers 1 \
--crop_size 256 \
--no_gpu "false" \
--rainaug "false" \



