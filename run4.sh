CUDA_VISIBLE_DEVICES=0,2 python pdseg/train.py --cfg ./configs/sf-hr48-agg-4.yaml --do_eval | tee -a ./runs/agg4
CUDA_VISIBLE_DEVICES=0,2 python pdseg/train.py --cfg ./configs/sf-hr48-agg-5.yaml --do_eval | tee -a ./runs/agg5
CUDA_VISIBLE_DEVICES=0,2 python pdseg/train.py --cfg ./configs/sf-hr48-agg-6.yaml --do_eval | tee -a ./runs/agg6