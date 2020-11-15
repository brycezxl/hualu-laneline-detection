CUDA_VISIBLE_DEVICES=0,1 python pdseg/train.py --cfg ./configs/sf-hr48-weight-2.yaml --do_eval | tee -a ./runs/w2
CUDA_VISIBLE_DEVICES=0,1 python pdseg/train.py --cfg ./configs/sf-hr48-weight-3.yaml --do_eval | tee -a ./runs/w3

# CUDA_VISIBLE_DEVICES=0 python pdseg/vis.py --cfg ./configs/sf-hr18-city-1.yaml --use_gpu