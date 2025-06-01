# flutter_classifier

### Требования

- Python 3.12
- [Poetry](https://python-poetry.org/docs/) for environment management

### Setup

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

1. Отредактируйте `/config/Train/Train.yaml`
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

## 3. Infer

1. Отредактируйте `/configs/Infer/Infer.yaml`, укажите чекпойнт и изображение,
   которе хотите использовать. Если не указать чекпойнт, то автоматически будет
   выбран лучший в папке.
2. Запустите
3. ```
   poetry run python infer.py
   ```
