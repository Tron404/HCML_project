#!/bin/bash

if pgrep -x "tensorboard" > /dev/null; then
    echo "Tensorboard is already running"
else
    tensorboard --logdir=regressor_bert/runs/ --host localhost --port 8888 --reload_interval 0.001 &
    sleep 1
    firefox 127.0.0.1:8888
fi

# keep last 10 runs; -gt=greater than
files=(regressor_bert/runs/*)
if [[ ${#files[@]} -gt 10 ]]; then
    cd regressor_bert/runs/
    ls -1 | head -n -10 | xargs -r rm -r
    cd ../..
fi

python bert_training.py \
    --agg "mean" --freeze "all_but_last_1" --hidden -1 -2 \
    --datasize 10000 --batches 10 --gpu true \
    --optimizer "adamw" --lr 0.005144591620070551 --w_decay 0.042581857401925625 \
    --epochs 25 --loss_func "rmse"\
