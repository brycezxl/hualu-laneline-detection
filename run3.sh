CUDA_VISIBLE_DEVICES=0 python pdseg/train.py --cfg ./configs/sf-hr18-size-12.yaml --do_eval | tee -a ./runs/size12
CUDA_VISIBLE_DEVICES=0 python pdseg/train.py --cfg ./configs/sf-hr18-size-13.yaml --do_eval | tee -a ./runs/size13

# CUDA_VISIBLE_DEVICES=0 python pdseg/vis.py --cfg ./configs/sf-hr18-city-1.yaml --use_gpu