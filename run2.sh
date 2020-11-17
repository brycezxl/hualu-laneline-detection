CUDA_VISIBLE_DEVICES=2 python pdseg/train.py --cfg ./configs/sf-hr18-loss-2.yaml --do_eval | tee -a ./runs/lovaz2
CUDA_VISIBLE_DEVICES=2 python pdseg/train.py --cfg ./configs/sf-hr18-loss-3.yaml --do_eval | tee -a ./runs/lovaz3
