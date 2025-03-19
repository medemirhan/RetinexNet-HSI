Supervised.
RetinexNet paper'ının RGB'den HSI'a uyarlanmış hali.
Network yapısı değiştirilmedi, sadece HSI'ı input alacak hale getirildi.
nn env'inde çalışıyor.

Sonuçlar aşağıdaki şekilde alındı:

python main.py --phase=train --epoch=2000 --eval_every_epoch=200
python main.py --phase=test

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