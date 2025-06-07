# flutter_classifier

### Описание

Проект выполняет классификацию бабочек и моли по изображениям. Проект загружает
копию датасета из Яндекс диска при помощи `download_data.py`, обучает модель в
`train.py`, также реализован модуль `infer.py`, позволяющий выполнять
классификацию при помощи обученной модели.

### Датасет

Использован датасет
[Butterfly &amp; Moths](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species),
содержащий 13594 изображений насекомых разных видов. Датасет разделен на 3
части: train (12594 изображений), validation (500 изображений), test (500
изображений). Каждая часть содержит 100 папок для каждого из видов.

### Требования

- Python 3.12
- [Poetry](https://python-poetry.org/docs/) for environment management
- DVC (опционально)
-

### Запуск

1. Склонируйте репозиторий

```bash
git clone https://github.com/maximkes/flutter_classifier.git
cd flutter_classifier
```

2. Устанивте зависимости, используя Poetry:

```bash
poetry install
```

### 1. Data Loading

Датасет будет загружен из яндекс диска, это может занять.

```bash
poetry run python download_data.py
```

Сохраните в dvc

```bash
dvc push
```

## 2. Train

Для запуска обучения

1. (Опционально) Отредактируйте `/config/Train/Train.yaml`
2. Запустите сервер mlflow

```bash
mlflow server --host 127.0.0.1 --port 8080
```

или поставьте `configs['Train']['use_MLFlow'] = False`

3. Запустите процесс обучения

```bash
poetry run python train.py

```

Чекпойнты сохраняются по умолчанию в папку `checkpoint/`, onnx по умолчанию
сохраняется в файл `onnx_export/model.onnx`

4. (Опционально) Экспорт модели в tensort (необходим trtexec)

```bash
./convert_onnx_to_tensorrt.sh
```

или

```bash
./convert_onnx_to_tensorrt.sh /custom/path/model.onnx
```

## 3. Infer

1. Отредактируйте `/configs/Infer/Infer.yaml`, укажите чекпойнт и изображение,
   которе хотите использовать. Если не указать чекпойнт, то автоматически будет
   выбран лучший в папке.
2. Запустите
3. ```bash
   poetry run python infer.py
   ```
