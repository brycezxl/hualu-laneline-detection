CUDA_VISIBLE_DEVICES=2 python pdseg/train.py --cfg ./configs/zxl-hr48-11.yaml --do_eval | tee -a ./train_log/hr4811
CUDA_VISIBLE_DEVICES=2 python pdseg/train.py --cfg ./configs/zxl-hr48-12.yaml --do_eval | tee -a ./train_log/hr4812
CUDA_VISIBLE_DEVICES=2 python pdseg/train.py --cfg ./configs/zxl-hr48-13.yaml --do_eval | tee -a ./train_log/hr4813
