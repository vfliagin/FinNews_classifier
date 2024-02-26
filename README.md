# DistilBERT для классификации финансовых новостей
FinNews_classifier использует fine-tuned Distilbert для классификации финансовых новостей. Результат может быть использован для анализа фондового рынка или как ввод для другой модели, например предсказывающей стоимость ценных бумаг.

## Обучение
Для обучения используется датасет [financial_phrasebank](https://huggingface.co/datasets/financial_phrasebank). Данные несбалансированны, поэтому в качестве метрики используется F1 score. Для загрузки и
дообучения модели используется библиотека transformers.

## Использование
Библиотека transformers позволяет очень легко использовать модель
```python
from transformers import pipeline

text = "Shake Shack stock surges 26% on fourth-quarter profit, strong 2024 outlook"

classifier = pipeline("sentiment-analysis", model = path_to_saved_model)
classifier(text)

[{'label': 'positive', 'score': 0.9823926091194153}]
```
## Гиперпараметры
Catastrophic forgetting - проблема, заключающаяся в потере знаний модели при дообучении на новых данных. Избежать её можно используя более низкий learning rate. Для файнтьюнинга
трансформеров для классификации обычно достаточно небольшого количества эпох. Для точного определения гиперпараметров можно использовать Ray Tune

## Результаты
![Матрица ошибок](/Матрица_ошибок.png)

## Источники
[DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)

[How to Fine-Tune BERT for Text Classification?](https://arxiv.org/pdf/1905.05583.pdf)

[Hugging Face Distillbert article](https://huggingface.co/docs/transformers/model_doc/distilbert)

