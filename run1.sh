CUDA_VISIBLE_DEVICES=1,2 python pdseg/train.py --cfg ./configs/zxl-hr64-base-1.yaml --do_eval | tee -a ./train_log/641
CUDA_VISIBLE_DEVICES=1,2 python pdseg/train.py --cfg ./configs/zxl-hr64-base-2.yaml --do_eval | tee -a ./train_log/642
CUDA_VISIBLE_DEVICES=1,2 python pdseg/train.py --cfg ./configs/zxl-hr64-base-3.yaml --do_eval | tee -a ./train_log/643
