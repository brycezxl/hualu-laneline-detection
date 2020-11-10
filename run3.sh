CUDA_VISIBLE_DEVICES=0,1 python pdseg/train.py --cfg ./configs/sf-hr18-agg-1.yaml --do_eval | tee -a ./runs/agg1
CUDA_VISIBLE_DEVICES=0,1 python pdseg/train.py --cfg ./configs/sf-hr18-agg-2.yaml --do_eval | tee -a ./runs/agg2
CUDA_VISIBLE_DEVICES=0,1 python pdseg/train.py --cfg ./configs/sf-hr18-agg-3.yaml --do_eval | tee -a ./runs/agg3