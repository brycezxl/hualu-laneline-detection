CUDA_VISIBLE_DEVICES=0 python pdseg/train.py --cfg ./configs/sf-hr18-loss-2.yaml --do_eval | tee -a ./runs/laneloss2
CUDA_VISIBLE_DEVICES=0 python pdseg/train.py --cfg ./configs/sf-hr18-loss-3.yaml --do_eval | tee -a ./runs/laneloss3
