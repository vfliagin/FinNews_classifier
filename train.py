import os
import evaluate
import numpy as np
from ray import tune
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

class ClassifierTrainer():
    def __init__(self):
        
        self.path = os.getcwd()
        #Загрузка и разделение данных
        self.fin_news = load_dataset("financial_phrasebank", "default", cache_dir=self.path)
        self.fin_news = self.fin_news["train"].train_test_split(test_size=0.2)
        #Токенизация
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.tokenized_fin_news = self.fin_news.map(self.preprocess_function, batched=True)
        #Создание сборщика данных
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        self.label2id = {"negative": 0, "neutral": 1, "positive": 2}
        #Определение метрики
        self.accuracy = evaluate.load("f1")
        
    def preprocess_function(self, examples):
        return self.tokenizer(examples["sentence"], truncation=True)
    
    def compute_metrics(self, eval_pred):
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        return self.accuracy.compute(predictions=predictions, references=labels, average="weighted")

    def model_init(self):
        
        model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3, id2label=self.id2label, label2id=self.label2id)
        
        return model

    def hp_space_ray(trial):
        """Определение возможных значений гипепараметров для оптимизации"""
    
        return {
            "learning_rate": tune.loguniform(1e-5, 5e-5),
            "num_train_epochs": tune.choice(range(1, 6)),
            "seed": tune.choice(range(1, 41)),
            "per_device_train_batch_size": tune.choice([4, 8, 16]),
        }
    
    def optimize_the_hyperparams(self):
        """Поиск оптимального значения гиперпараметров"""

        training_args = TrainingArguments(
            output_dir = self.path + 'HPO/',
            evaluation_strategy="steps",
            eval_steps=500,
            load_best_model_at_end=True,
            push_to_hub=False)

        trainer = Trainer(
            model_init=self.model_init,
            args=training_args,
            train_dataset=self.tokenized_fin_news["train"],
            eval_dataset=self.tokenized_fin_news["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics)

        trainer.hyperparameter_search(
            direction="maximize",
            backend="ray",
            n_trials=6,
            hp_space=self.hp_space_ray)
    
    def train_the_classifier(self):
        """Обучение модели"""
        
        training_args = TrainingArguments(
            output_dir = self.path,
            learning_rate = 2e-5,
            per_device_train_batch_size = 16,
            per_device_eval_batch_size = 16,
            num_train_epochs = 4,
            weight_decay = 0.01,
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            load_best_model_at_end = True,
            push_to_hub=False)

        trainer = Trainer(
            model_init=self.model_init,
            args=training_args,
            train_dataset = self.tokenized_fin_news["train"],
            eval_dataset = self.tokenized_fin_news["test"],
            tokenizer = self.tokenizer,
            data_collator = self.data_collator,
            compute_metrics = self.compute_metrics)
        
        trainer.train()
        
        trainer.save_model(self.path + 'saved_model/')

if __name__ == '__main__':
    trainer = ClassifierTrainer()
    trainer.train_the_classifier()
