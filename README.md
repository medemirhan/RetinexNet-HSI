## HSI implementation of RetinexNet (RGB) paper.
- Supervised.
- Network structure of RetinexNet preserved.
- nn conda env

### Training usage ###
```shell
python main.py --phase=train --epoch=2000 --eval_every_epoch=200
```

### Testing usage ###
```shell
python main.py --phase=test
```