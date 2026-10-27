import evaluate
import itertools
import json
import numpy as np
import torch
from datasets import Dataset
from seqeval.metrics import classification_report
from tabulate import tabulate
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer
from utilities import read_conll_file



MODELS = [
        "distilbert/distilbert-base-cased",
        "google-bert/bert-base-cased",
        "eventdata-utd/conflibert-named-entity-recognition"
        ]

TRAIN_DATA = "data/data.txt"
GOLD_LABEL = "data/gold_test.txt"
IGNORE_LOSS = -100

model_performance = {}

# process data
train_words, train_tags = read_conll_file(TRAIN_DATA)
gold_words, gold_tags = read_conll_file(GOLD_LABEL)

unique_tags = sorted(list(set(tag for doc in train_tags for tag in doc)))

if "O" in unique_tags:
    unique_tags.remove("O")

label_list = ["O"] + unique_tags

# Build label maps
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# build data dicts
data_dict = {
            'tokens': train_words,
            'ner_tags':train_tags
}

gold_data_dict = {
            'tokens': gold_words,
            'ner_tags':gold_tags
}

dataset = Dataset.from_dict(data_dict)
gold_dataset = Dataset.from_dict(gold_data_dict)

#compute metrics
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p

    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (-100) and convert back to strings
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    
    return {
        "precision": f"{results['overall_precision']:.4}",
        "recall": f"{results['overall_recall']:.4}",
        "f1":f"{results['overall_f1']:.4}" ,
        "accuracy": f"{results['overall_accuracy']:.4}",
    }
    
'''
Experiments on three different models
'''

def experiment_1():
    model_name = "distilbert/distilbert-base-cased"
    # tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # define tokenize and align here to pass to map
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
                                    examples["tokens"], 
                                    truncation=True, 
                                    is_split_into_words=True
                                    )
        labels = []
        
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None: # Special tokens like [CLS]
                    label_ids.append(IGNORE_LOSS)
                elif word_idx != previous_word_idx: # First subword of a word
                    label_ids.append(label2id[label[word_idx]])
                else: # Subsequent subwords
                    label_ids.append(IGNORE_LOSS) 
                    
                previous_word_idx = word_idx
                
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # pass tokenize_and_align to map
    tokenized_ds = dataset.map(tokenize_and_align_labels, batched=True,load_from_cache_file=False)
    tokenized_gold_ds = gold_dataset.map(tokenize_and_align_labels, batched=True, load_from_cache_file=False)

    full_ds = tokenized_ds.train_test_split(test_size=0.3)
    train_ds = full_ds['train']
    eval_split = full_ds['test'].train_test_split(test_size=0.5)
    dev_ds =eval_split['train']
    eval_ds = eval_split['test']

    # Init model
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, 
        num_labels=len(unique_tags),
        id2label=id2label,
        label2id=label2id
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    cleaned_model_name = model_name.replace("/","-")
    training_args = TrainingArguments(
        output_dir=cleaned_model_name,
        eval_strategy="epoch",          
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=30,           
        weight_decay=0.01,
        logging_steps=10,
        push_to_hub=False,            
    )

    # begin model here 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds, 
        eval_dataset=dev_ds,       
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # finetune the model
    trainer.train()

    # evaluate against a test split from the training data:
    synthetic_results = trainer.evaluate(eval_dataset=eval_ds)
    
    # evaluate against my gold_label dataset
    gold_results = trainer.evaluate(eval_dataset=tokenized_gold_ds)

    # display evaluation metrics
    print(f"Synthetic Data Finetuned results on {model_name}:")
    for i,v in synthetic_results.items():
        print(f"{i}: {v}")
        
    print(f"Gold Label Data Finetuned results on {model_name}:")
    for i,v in gold_results.items():
        print(f"{i}: {v}")

    model_performance[cleaned_model_name] = {
        "synthetic": {
            "f1": float(synthetic_results['eval_f1']),
            "accuracy": float(synthetic_results['eval_accuracy'])
        },
        "gold": {
            "f1": float(gold_results['eval_f1']),
            "accuracy": float(gold_results['eval_accuracy'])
        }
    }
    
def experiment_2():
    model_name = "google-bert/bert-base-cased"
    # tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # define tokenize and align here to pass to map
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
                                    examples["tokens"], 
                                    truncation=True, 
                                    is_split_into_words=True
                                    )
        labels = []
        
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None: # Special tokens like [CLS]
                    label_ids.append(IGNORE_LOSS)
                elif word_idx != previous_word_idx: # First subword of a word
                    label_ids.append(label2id[label[word_idx]])
                else: # Subsequent subwords
                    label_ids.append(IGNORE_LOSS) 
                    
                previous_word_idx = word_idx
                
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # pass tokenize_and_align to map
    tokenized_ds = dataset.map(tokenize_and_align_labels, batched=True,load_from_cache_file=False)
    tokenized_gold_ds = gold_dataset.map(tokenize_and_align_labels, batched=True, load_from_cache_file=False)

    full_ds = tokenized_ds.train_test_split(test_size=0.3)
    train_ds = full_ds['train']
    eval_split = full_ds['test'].train_test_split(test_size=0.5)
    dev_ds =eval_split['train']
    eval_ds = eval_split['test']
    
    # Init model
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, 
        num_labels=len(unique_tags),
        id2label=id2label,
        label2id=label2id
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    cleaned_model_name = model_name.replace("/","-")
    training_args = TrainingArguments(
        output_dir=cleaned_model_name,
        eval_strategy="epoch",          
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=30,           
        weight_decay=0.01,
        logging_steps=10,
        push_to_hub=False,            
    )

    # begin model here 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds, 
        eval_dataset=dev_ds,       
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # finetune the model
    trainer.train()

    # evaluate against a test split from the training data:
    synthetic_results = trainer.evaluate(eval_dataset=eval_ds)
    
    # evaluate against my gold_label dataset
    gold_results = trainer.evaluate(eval_dataset=tokenized_gold_ds)

    # display evaluation metrics
    print(f"Synthetic Data Finetuned results on {model_name}:")
    for i,v in synthetic_results.items():
        print(f"{i}: {v}")
        
    print(f"Gold Label Data Finetuned results on {model_name}:")
    for i,v in gold_results.items():
        print(f"{i}: {v}")

    model_performance[cleaned_model_name] = {
        "synthetic": {
            "f1": float(synthetic_results['eval_f1']),
            "accuracy": float(synthetic_results['eval_accuracy'])
        },
        "gold": {
            "f1": float(gold_results['eval_f1']),
            "accuracy": float(gold_results['eval_accuracy'])
        }
    }

def experiment_3():
    model_name = "eventdata-utd/conflibert-named-entity-recognition"
    # tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # define tokenize and align here to pass to map
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
                                    examples["tokens"], 
                                    truncation=True, 
                                    is_split_into_words=True
                                    )
        labels = []
        
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None: # Special tokens like [CLS]
                    label_ids.append(IGNORE_LOSS)
                elif word_idx != previous_word_idx: # First subword of a word
                    label_ids.append(label2id[label[word_idx]])
                else: # Subsequent subwords
                    label_ids.append(IGNORE_LOSS) 
                    
                previous_word_idx = word_idx
                
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # pass tokenize_and_align to map
    tokenized_ds = dataset.map(tokenize_and_align_labels, batched=True,load_from_cache_file=False)
    tokenized_gold_ds = gold_dataset.map(tokenize_and_align_labels, batched=True, load_from_cache_file=False)

    full_ds = tokenized_ds.train_test_split(test_size=0.3)
    train_ds = full_ds['train']
    eval_split = full_ds['test'].train_test_split(test_size=0.5)
    dev_ds =eval_split['train']
    eval_ds = eval_split['test']


    # Init model
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, 
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    cleaned_model_name = model_name.replace("/","-")
    training_args = TrainingArguments(
        output_dir=cleaned_model_name,
        eval_strategy="epoch",          
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=30,           
        weight_decay=0.01,
        logging_steps=10,
        push_to_hub=False,            
    )

    # begin model here 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds, 
        eval_dataset=dev_ds,       
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # finetune the model
    trainer.train()

    # evaluate against a test split from the training data:
    synthetic_results = trainer.evaluate(eval_dataset=eval_ds)
    
    # evaluate against my gold_label dataset
    gold_results = trainer.evaluate(eval_dataset=tokenized_gold_ds)

    # display evaluation metrics
    print(f"Synthetic Data Finetuned results on {model_name}:")
    for i,v in synthetic_results.items():
        print(f"{i}: {v}")
        
    print(f"Gold Label Data Finetuned results on {model_name}:")
    for i,v in gold_results.items():
        print(f"{i}: {v}")

    model_performance[cleaned_model_name] = {
        "synthetic": {
            "f1": float(synthetic_results['eval_f1']),
            "accuracy": float(synthetic_results['eval_accuracy'])
        },
        "gold": {
            "f1": float(gold_results['eval_f1']),
            "accuracy": float(gold_results['eval_accuracy'])
        }
    }
    
    
def grid_search():
    best_model = 'eventdata-utd/conflibert-named-entity-recognition'
    
    parameter_grid = {
        "batch_size": [16, 32, 64],
        "epochs":[5,10,15,20,25,30],
        "learning_rate": [5e-5,4e-5,3e-5,2e-5,1e-5],
        "weight_decay": [0.0,0.01,0.1],
    }

    best_f1 = 0.0
    best_parameters = {}
    
    keys,values = zip(*parameter_grid.items())
    grid_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    tokenizer = AutoTokenizer.from_pretrained(best_model, use_fast=False)

    # define tokenize and align here to pass to map
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
                                    examples["tokens"], 
                                    truncation=True, 
                                    is_split_into_words=True
                                    )
        labels = []
        
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None: # Special tokens like [CLS]
                    label_ids.append(IGNORE_LOSS)
                elif word_idx != previous_word_idx: # First subword of a word
                    label_ids.append(label2id[label[word_idx]])
                else: # Subsequent subwords
                    label_ids.append(IGNORE_LOSS) 
                    
                previous_word_idx = word_idx
                
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # pass tokenize_and_align to map
    tokenized_ds = dataset.map(tokenize_and_align_labels, batched=True,load_from_cache_file=False)
    tokenized_gold_ds = gold_dataset.map(tokenize_and_align_labels, batched=True, load_from_cache_file=False)

    full_ds = tokenized_ds.train_test_split(test_size=0.3)
    train_ds = full_ds['train']
    eval_split = full_ds['test'].train_test_split(test_size=0.5)
    dev_ds =eval_split['train']
    eval_ds = eval_split['test']

    # config = BertConfig.from_pretrained(best_model, num_labels=len(label2id))
    # Init model
    model = AutoModelForTokenClassification.from_pretrained(
        best_model, 
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    cleaned_model_name = best_model.replace("/","-")
    
    for config in grid_configs:
        torch.cuda.empty_cache()
        training_args = TrainingArguments(
            output_dir=cleaned_model_name,
            eval_strategy="epoch",          
            learning_rate=config["learning_rate"],
            per_device_train_batch_size=config["batch_size"],
            num_train_epochs=config["epochs"],           
            weight_decay=config["weight_decay"],
            logging_steps=10,
            push_to_hub=False          
        )

        # begin model here 
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds, 
            eval_dataset=dev_ds,       
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        # finetune the model
        trainer.train()

        # evaluate against my gold_label dataset
        results = trainer.evaluate(eval_dataset=tokenized_gold_ds)
        current_f1 = float(results["eval_f1"])
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_params = config
            
    with open("best_hyperparameters.json", "w") as f:
        json.dump({"best_params": best_params, "best_f1": best_f1}, f, indent=4)
        
    print(best_parameters)


if __name__ == "__main__":
    # experiment_1()
    # experiment_2()
    # experiment_3()

    grid_search()