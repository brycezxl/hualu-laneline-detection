CUDA_VISIBLE_DEVICES=2 python pdseg/train.py --cfg ./configs/sf-hr18-city-1.yaml --do_eval | tee -a ./runs/city1
CUDA_VISIBLE_DEVICES=2 python pdseg/train.py --cfg ./configs/sf-hr18-city-2.yaml --do_eval | tee -a ./runs/city2
CUDA_VISIBLE_DEVICES=2 python pdseg/train.py --cfg ./configs/sf-hr18-city-3.yaml --do_eval | tee -a ./runs/city3

# CUDA_VISIBLE_DEVICES=0 python pdseg/vis.py --cfg ./configs/sf-hr18-city-1.yaml --use_gpu