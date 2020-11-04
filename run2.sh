CUDA_VISIBLE_DEVICES=1 python pdseg/train.py --cfg ./configs/sf-hr18-image-1.yaml --do_eval | tee -a ./train_log/img1
CUDA_VISIBLE_DEVICES=1 python pdseg/train.py --cfg ./configs/sf-hr18-image-2.yaml --do_eval | tee -a ./train_log/img2
CUDA_VISIBLE_DEVICES=1 python pdseg/train.py --cfg ./configs/sf-hr18-image-3.yaml --do_eval | tee -a ./train_log/img3
