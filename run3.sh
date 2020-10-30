CUDA_VISIBLE_DEVICES=0 python pdseg/train.py --cfg ./configs/zxl-hr48-14.yaml --do_eval | tee -a ./train_log/hr4814
CUDA_VISIBLE_DEVICES=0 python pdseg/train.py --cfg ./configs/zxl-hr48-15.yaml --do_eval | tee -a ./train_log/hr4815
