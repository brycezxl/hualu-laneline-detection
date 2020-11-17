CUDA_VISIBLE_DEVICES=0 python pdseg/train.py --cfg ./configs/sf-hr18-agg-1.yaml --do_eval | tee -a ./runs/18agg11
CUDA_VISIBLE_DEVICES=0 python pdseg/train.py --cfg ./configs/sf-hr18-agg-2.yaml --do_eval | tee -a ./runs/18agg2
