# conda activate hmcl_proj
python bert_training.py \
    --agg "mean" --freeze true --dense "128,128" --hidden -1 \
    --datasize 750 --batches 1 --gpu false \
    --optimizer "adamw" --lr 0.001 --w_decay 0.01 \
    --epochs 10  --loss_func "rmse" 

# conda deactivate
