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
трансформеров для классификации обычно достаточно небольшого количества эпох. Для точного определения гиперпараметров был использован Ray Tune. Перебор гиперпараметров
из интревала [1e-4, 1e-2] для learning rate дал лучший F1 score у модели с самым низким значением. Модели с более высоким learning rate оказались значительно хуже. Было принято решение использовать learning rate 2e-5 и 4 эпохи
```python
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Trial name               status         learning_rate     num_train_epochs     seed     ..._train_batch_size     iter     total time (s)     eval_loss     eval_f1     eval_runtime     ...amples_per_second |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| _objective_ba55b_00000   TERMINATED       0.000561152                    5       15                       16        2            123.371      0.921221    0.446235           1.744                   556.194 |
| _objective_ba55b_00001   TERMINATED       0.00362562                     5       39                        8        4            207.722      0.922004    0.446235           1.8182                  533.505 |
| _objective_ba55b_00002   TERMINATED       0.000205111                    3       11                        4        5            209.663      0.921161    0.446235           1.7556                  552.527 |
| _objective_ba55b_00003   TERMINATED       0.00159305                     3       22                        4        5            230.859      0.929211    0.446235           1.8004                  538.755 |
| _objective_ba55b_00004   TERMINATED       0.00870602                     4       30                        8        3            166.807      0.921125    0.446235           1.6671                  581.858 |
| _objective_ba55b_00005   TERMINATED       0.000100359                    4       21                        4        7            392.954      1.14842     0.807193           1.7988                  539.261 |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

BestRun(run_id='ba55b_00005', objective=0.8071927884678695, hyperparameters={'learning_rate': 0.00010035927878780932, 'num_train_epochs': 4, 'seed': 21, 'per_device_train_batch_size': 4}, run_summary=<ray.tune.analysis.experiment_analysis.ExperimentAnalysis object at 0x7c90a8dc5d20>)
```
## Результаты
F1 score получившейся модели 0.9229

![Матрица ошибок](/Матрица_ошибок.png)

## Источники
[DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)

[How to Fine-Tune BERT for Text Classification?](https://arxiv.org/pdf/1905.05583.pdf)

[Hugging Face Distillbert article](https://huggingface.co/docs/transformers/model_doc/distilbert)

[Using hyperparameter-search in Trainer](https://discuss.huggingface.co/t/using-hyperparameter-search-in-trainer/785)
