CUDA_VISIBLE_DEVICES=0 python pdseg/train.py --cfg ./configs/sf-hr18-city-1.yaml --do_eval | tee -a ./train_log/city1
CUDA_VISIBLE_DEVICES=0 python pdseg/train.py --cfg ./configs/sf-hr18-city-2.yaml --do_eval | tee -a ./train_log/city2
CUDA_VISIBLE_DEVICES=0 python pdseg/train.py --cfg ./configs/sf-hr18-city-3.yaml --do_eval | tee -a ./train_log/city3
