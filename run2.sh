CUDA_VISIBLE_DEVICES=0,1 python pdseg/train.py --cfg ./configs/sf-hr48-final-1.yaml --do_eval | tee -a ./runs/final1
CUDA_VISIBLE_DEVICES=0,1 python pdseg/train.py --cfg ./configs/sf-hr48-final-2.yaml --do_eval | tee -a ./runs/final2
