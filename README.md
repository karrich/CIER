# CIER(Coherency-Improved Explainable Recommendation via Large Language Model)

## Datasets to [download](https://lifehkbueduhk-my.sharepoint.com/:f:/g/personal/16484134_life_hkbu_edu_hk/Eln600lqZdVBslRwNcAJL5cBarq6Mt8WzDKpkq1YCqQjfQ?e=cISb1C)
- TripAdvisor Hong Kong
- Amazon Movies & TV
- Yelp 2019

## Training
Below are examples of how to training CIER.
```
python main.py --dataset_name Yelp

python main.py --dataset_name Amazon/MoviesAndTV

python main.py --dataset_name TripAdvisor
```
## Only test
```
python main.py --dataset_name Yelp --only_eval
```
## Code dependencies
- Python 3.9
- PyTorch 2.2.2
- transformers 4.37.2
- peft 0.3.0
- accelerate 0.28.0
