CUDA_VISIBLE_DEVICES=0 python pdseg/train.py --cfg ./configs/sf-hr18-step-1.yaml --do_eval | tee -a ./runs/step1
CUDA_VISIBLE_DEVICES=0 python pdseg/train.py --cfg ./configs/sf-hr18-step-2.yaml --do_eval | tee -a ./runs/step2
CUDA_VISIBLE_DEVICES=0 python pdseg/train.py --cfg ./configs/sf-hr18-step-3.yaml --do_eval | tee -a ./runs/step3
