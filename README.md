# Enhancing Descriptive Image Captioning with Natural Language Inference

ACL 2021

## Requirements

- Python 3.6
- Java 1.8.0
- PyTorch 1.0
- cider (https://github.com/ruotianluo/cider/tree/dbb3960165d86202ed3c417b412a000fc8e717f3) and replace "cider/pyciderevalcap/ciderD"  with the same subfolder submitted here
- coco-caption (https://github.com/ruotianluo/coco-caption/tree/dda03fc714f1fcc8e2696a8db0d469d99b881411)
- tensorboardX


## Training 

### Prepare data

See Most details in `data/README.md`.

Download nli data [here](https://drive.google.com/drive/folders/1wD8ThjwQknvOiRlxRvcVXgflZ69CXnsW?usp=sharing). 

1. coco_nli_new.json is the inference result between multiple references.

2. nli_dist_mle and nli_dist_rl are output of page-rank algorithm.

   ```shell
   cd experiment
   python analysis.py
   ```

### Start training

```bash
$ CUDA_VISIBLE_DEVICES=0 ./train_aoa.sh
```


### Evaluation
You may use trained models here [google drive](https://drive.google.com/drive/folders/1lAvHI4Jek2Avbv2DLlPI0_btRQ62ZjZG?usp=sharing)

```bash
$ CUDA_VISIBLE_DEVICES=0 ./test-best.sh
```


## Reference

If you find this repo helpful, please consider citing:

```
@inproceedings{shi-etal-2021-enhancing,
    title = "Enhancing Descriptive Image Captioning with Natural Language Inference",
    author = "Shi, Zhan  and
      Liu, Hui  and
      Zhu, Xiaodan",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-short.36",
    doi = "10.18653/v1/2021.acl-short.36",
    pages = "269--277",
}

```

## Acknowledgements

This repository is based on [AoANet](https://github.com/husthuaan/AoANet), and you may refer to it for more details about the code.
