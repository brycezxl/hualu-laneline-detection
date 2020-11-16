CUDA_VISIBLE_DEVICES=0,1 python pdseg/train.py --cfg ./configs/sf-hr64-agg-1.yaml --do_eval | tee -a ./runs/641
CUDA_VISIBLE_DEVICES=0,1 python pdseg/train.py --cfg ./configs/sf-hr64-agg-2.yaml --do_eval | tee -a ./runs/642
CUDA_VISIBLE_DEVICES=0,1 python pdseg/train.py --cfg ./configs/sf-hr64-agg-3.yaml --do_eval | tee -a ./runs/643

# CUDA_VISIBLE_DEVICES=0 python pdseg/vis.py --cfg ./configs/sf-hr18-city-1.yaml --use_gpu