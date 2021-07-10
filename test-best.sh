#!/bin/sh
CUDA_VISIBLE_DEVICES=1 python eval.py --dump_images 0 --dump_json 1 --num_images -1 --model loge/log_nli_aoa_rl/model-best.pth --infos_path loge/log_nli_aoa_rl/infos_nli_aoa.pkl --language_eval 1 --beam_size 2 --batch_size 100 --split test --retrieval 1
