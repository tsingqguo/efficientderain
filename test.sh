for i in $(seq 1 1 1)
do
    echo "dealing with ${i}:";
    python ./validation.py \
    --load_name "./models/v3_SPA/v3_SPA.pth" \
    --save_name "./results/results_tmp" \
    --baseroot "./datasets/SPA/testing" ;
done