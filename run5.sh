CUDA_VISIBLE_DEVICES=0 python pdseg/train.py --cfg ./configs/sf-hr18-norm-4.yaml --do_eval | tee -a ./runs/norm4
CUDA_VISIBLE_DEVICES=0 python pdseg/train.py --cfg ./configs/sf-hr18-norm-3.yaml --do_eval | tee -a ./runs/norm3

# CUDA_VISIBLE_DEVICES=0 python pdseg/vis.py --cfg ./configs/sf-hr18-city-1.yaml --use_gpu