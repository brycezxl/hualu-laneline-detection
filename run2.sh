CUDA_VISIBLE_DEVICES=2 python pdseg/train.py --cfg ./configs/zxl-hr48-7.yaml --do_eval | tee -a ./train_log/hr487
CUDA_VISIBLE_DEVICES=2 python pdseg/train.py --cfg ./configs/zxl-hr48-8.yaml --do_eval | tee -a ./train_log/hr488
