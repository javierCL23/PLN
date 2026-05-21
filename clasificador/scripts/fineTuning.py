import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

from src.config import PATH_DATA, PATH_MODELS

MODEL_NAME = "distilbert-base-uncased"
PATH_MODEL_OUTPUT = PATH_MODELS / MODEL_NAME
TEMP_DIR = Path("/tmp/resultados_temporales")

# FUNCIÓN DE MÉTRICAS PARA EVALUAR DURANTE EL ENTRENAMIENTO
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# MAIN
def main():
    print("- Iniciando Fine-Tuning de Transformers...")
    
    # CARGA Y PREPARACIÓN DE DATOS
    print("- Cargando datasets...")
    # Asegúrate de cargar los datos
    df_train = pd.read_json(PATH_DATA / "train_preprocesado.jsonl",lines=True)
    df_train['label'] = df_train['label'] - 1
    
    # Convertimos todo el dataframe a formato Dataset de Hugging Face
    dataset_full = Dataset.from_pandas(df_train[['desc_minimal', 'label']])
    
    dataset_split = dataset_full.train_test_split(test_size=0.2, seed=42)
    
    dataset_train = dataset_split['train']
    dataset_val   = dataset_split['test']

    print(f"    Ejemplos para entrenar: {len(dataset_train)}")
    print(f"    Ejemplos para validar:  {len(dataset_val)}")

    # TOKENIZACIÓN
    print(f"- Cargando tokenizador de {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        # Truncar a 128 para mayor velocidad en el fine-tunning
        return tokenizer(examples["desc_minimal"], truncation=True, max_length=128)

    print("- Tokenizando...")
    tokenized_train = dataset_train.map(tokenize_function, batched=True)
    tokenized_val   = dataset_val.map(tokenize_function, batched=True)

    # INICIALIZACIÓN DEL MODELO
    print("- Descargando/Cargando el modelo base...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # CONFIGURACIÓN DEL ENTRENAMIENTO
    training_args = TrainingArguments(
        output_dir=TEMP_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"
    )

    # ENTRENAMIENTO
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print("- Iniciando el Fine-Tuning...")
    trainer.train()

    # GUARDADO (Modelo + Tokenizador)
    print(f"Guardando el modelo definitivo en: {PATH_MODEL_OUTPUT}")
    PATH_MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    # Pesos del modelo
    trainer.save_model(PATH_MODEL_OUTPUT)
    # Vocabulario del tokenizador para poder leerlo en Main
    tokenizer.save_pretrained(PATH_MODEL_OUTPUT)


    # Limpieza de los resultados intermedios
    if TEMP_DIR.exists() and TEMP_DIR.is_dir():
        import shutil
        shutil.rmtree(TEMP_DIR)

    print("- Fine-Tuning finalizado.")

if __name__ == "__main__":
    main()