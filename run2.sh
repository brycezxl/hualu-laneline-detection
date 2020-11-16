CUDA_VISIBLE_DEVICES=0 python pdseg/train.py --cfg ./configs/sf-hr18-loss-1.yaml --do_eval | tee -a ./runs/laneloss1
CUDA_VISIBLE_DEVICES=0 python pdseg/train.py --cfg ./configs/sf-hr18-agg.yaml --do_eval | tee -a ./runs/18agg
