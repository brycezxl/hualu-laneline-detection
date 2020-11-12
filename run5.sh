CUDA_VISIBLE_DEVICES=0,1 python pdseg/train.py --cfg ./configs/sf-hr48-agg-8.yaml --do_eval | tee -a ./runs/agg8
CUDA_VISIBLE_DEVICES=0,1 python pdseg/train.py --cfg ./configs/sf-hr48-agg-9.yaml --do_eval | tee -a ./runs/agg9

# CUDA_VISIBLE_DEVICES=0 python pdseg/vis.py --cfg ./configs/sf-hr18-city-1.yaml --use_gpu