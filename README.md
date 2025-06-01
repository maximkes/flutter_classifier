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

Датасет будет загружен из google диска, это займет несколько минут.

```bash
poetry run python download_data.py
```

Сохраните в dvc

```bash
dvc push
```

## 2. Train

Для запуска обучения

1. Отредактируйте config
2. Запустите сервер mlflow

```bash
mlflow server --host 127.0.0.1 --port 8080
```

или поставьте `config['Train']['use_MLFlow'] = False`

3. Запустите процесс обучения

```bash
poetry run python train.py
```

## Infer

After training the model, you can use it for inference on new data. Set a path
to test data in hydra configuration, the same as in train.

### Running Inference

```bash
poetry run python3 -m wikics_segment_prediction.infer
```

This will output predictions for each class per node.

## Contact

Project author: [sergstan1](https://github.com/sergstan1)
