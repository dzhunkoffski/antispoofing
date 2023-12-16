# antispoofing
DL-AUDIO homework

## Get dataset:
```bash
скачай с каггла
```

## Train model
python train.py --config /kaggle/working/antispoofing/src/configs/rawnet2.json

## Test model
python test.py --config src/configs/rawnet2.json --resume checkpoint/rawnet2/checkpoint-epoch62.pth

файл `kaggle_notebook.ipynb` содержит 