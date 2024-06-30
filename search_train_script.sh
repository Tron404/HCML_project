#!/bin/bash
trap "kill 0" EXIT

options="d:n:"
while getopts ${options} opt; do
    case ${opt} in
        d) 
            dashboard=${OPTARG}
            if [ $dashboard != "tf" ] && [ $dashboard != "optuna" ]; then
                echo "ERROR: Please provide a correct dashboard name (tf/optuna) or remove the -d flag"
                exit 1  
            fi 
            ;;
        n)
            study_name=${OPTARG}
            ;;
        ?)
            echo "ERROR: Unknown flag option"
            exit 1
            ;;
    esac
done

# display results in a tensorflow dashboard in browser
if [ $dashboard == "tf" ]; then
    if pgrep -x "tensorboard" > /dev/null; then
        echo "Tensorboard is already running"
    else
        tensorboard --logdir=regressor_bert/runs/ --host localhost --port 8888 --reload_interval 0.001 &
        sleep 1
        firefox 127.0.0.1:8888
    fi
    # keep last 10 runs; -gt=greater than
    limit=10
    files=(regressor_bert/runs/*)
    if [[ ${#files[@]} -gt $limit ]]; then
        cd regressor_bert/runs/
        ls -1 | head -n -$limit | xargs -r rm -r
        cd ../..
    fi
fi

python hyperparameter_search.py \
    --agg "mean"\
    --datasize 1500 --batches 14 --gpu true \
    --optimizer "adamw" \
    --epochs 25 --loss_func "rmse" --num_trials 150 --study_name $study_name & process_pid=$!

# display results in optuna dashboard
if [ $dashboard == "optuna" ]; then
    if pgrep -x "optuna-dashboar" > /dev/null; then
        echo "Optuna is already running"
    else
        optuna-dashboard sqlite:///HPO.sqlite3 &
        sleep 1
        firefox 127.0.0.1:8080
    fi
fi

wait
