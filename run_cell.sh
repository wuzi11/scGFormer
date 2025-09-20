export SCIPY_ARRAY_API=1
python main.py --dataset Baron --data_dir ./dataset \
        --use_HVG \
        --epochs 40 \
        --loss ce \
        --device 0 \
        --use_performer \
        --use_gat \
        --use_knn \
        --seed 2 \
        --dynamic_graph \








