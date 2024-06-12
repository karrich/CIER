# ERRB (xxxx)

## Paper
- Shijie Liu, Ruixin Ding, xx xx, Wei Zhang. [name](https:xx). xxxx (EMNLP ing), 2024.

## Datasets to [download](https://lifehkbueduhk-my.sharepoint.com/:f:/g/personal/16484134_life_hkbu_edu_hk/Eln600lqZdVBslRwNcAJL5cBarq6Mt8WzDKpkq1YCqQjfQ?e=cISb1C)
- TripAdvisor Hong Kong
- Amazon Movies & TV
- Yelp 2019

## Training
Below are examples of how to training ERRB.
```
python main.py --delta 0.2 --dataset_name Yelp

python main.py --delta 0.6 --dataset_name Amazon/MoviesAndTV

python main.py --delta 0.2 --dataset_name TripAdvisor
```
## Only test
```
python main.py \xxxx
```
## Code dependencies
- Python 3.9
- PyTorch 2.2.2
- transformers 4.37.2
- peft 0.3.0
- accelerate 0.28.0

## If you use the code, please cite:
```
@inproceedings{ERRB
}
@inproceedings{ACL21-PETER,
	title={Personalized Transformer for Explainable Recommendation},
	author={Li, Lei and Zhang, Yongfeng and Chen, Li},
	booktitle={ACL},
	year={2021}
}
```
