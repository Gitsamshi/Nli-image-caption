#!/bin/sh
CUDA_VISIBLE_DEVICES=1 python eval_logloss.py --dump_images 0 --dump_json 1 --num_images -1 --model loge/log_aoanet_acl2021_basic/model-best.pth --infos_path loge/log_aoanet_acl2021_basic/infos_aoanet_acl2021_basic-best.pkl --language_eval 1 --beam_size 2 --batch_size 100
