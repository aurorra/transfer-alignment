import random
import os
import json
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["WANDB_DISABLED"] = "true"

import pandas as pd
import numpy as np
import torch
import gc
from transformers import Trainer, TrainingArguments, set_seed, DataCollatorWithPadding
from transformers.trainer_utils import get_last_checkpoint
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import seaborn as sns
from codecarbon import OfflineEmissionsTracker

from transfer.src.finetuning.model_utils import model_init


def set_seed_locally(seed=None):
    """
    Set random seeds for reproducible training. If seed is None, no seed settings are changed.

    :param seed: Random seed.
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


def empty_cache():
    """ Collect unused Python objects and release cached CUDA memory. """
    gc.collect()
    torch.cuda.empty_cache()


def compute_metrics(pred):
    """
    Computes performance metrics for the predictions made by a model.

    :param pred: An object containing the model's predictions and the corresponding true labels.
                 This object should have the following attributes:
                 - label_ids: an array of actual labels.
                 - predictions: a 2D array where each row represents the logit scores for each class.
    :return: A dictionary containing the performance metrics  accuracy, F1 score, recall, and precision, calculated
                based on the true labels and the model's predictions.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='binary', pos_label=1)
    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds, average='binary', pos_label=1)
    precision = precision_score(labels, preds, average='binary', pos_label=1)
    return {'accuracy': acc, 'f1': f1, 'recall': recall, 'precision': precision}


def build_confusion_matrix(dataset, predictions, path):
    """
    Builds and saves a confusion matrix as a heatmap from the given predictions and actual labels.

    This function calculates the confusion matrix using the actual labels and the predicted labels from a dataset.
    It then generates a heatmap visualization of the confusion matrix using seaborn and saves it as an image file.

    :param dataset: A dataset object containing 'label' as one of its keys, representing the actual labels.
    :param predictions: A list or array of predictions corresponding to the entries in the dataset.
    :param path: The file path where the confusion matrix image will be saved.
    """
    conf = confusion_matrix(dataset['label'], predictions)
    sns.heatmap(conf, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f'{path}/confusion_matrix.png')
    plt.clf()  # clear figure


def training_loop(model_ckpt, tokenizer, tokenized_hf_train, tokenized_hf_val, tokenized_hf_test, learning_rate,
                  num_epochs, batch_size, weight_decay, warmup_steps, warmup_ratio, output_path, num_inits, device,
                  lora=False, lora_r=4, lora_alpha=8, lora_dropout=0.0, lora_target_modules=None, lora_bias='none',
                  lora_modules_to_save=None):
    """
    Train and evaluate the model for multiple random initializations.
    """
    initializations_accuracies = {}
    for init in range(0, num_inits):
        empty_cache()

        # Use a stable output directory name when model_ckpt is a local path
        if ('/' in model_ckpt) or ('\\' in model_ckpt) or ('..' in model_ckpt):
            model_tag = 'bert-base-multilingual-cased'
        else:
            model_tag = model_ckpt

        # Ensure base directories exist
        init_base_dir = f'{output_path}/init_{init}'
        logs_dir = f'{init_base_dir}/logs'
        model_dir = f'{init_base_dir}/{model_tag}'
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # CodeCarbon tracker for this init
        tracker = OfflineEmissionsTracker(
            project_name=f'{model_tag}_finetuning',
            output_dir=model_dir,
            output_file='emissions.csv',
            save_to_file=True,
            measure_power_secs=10,  # sampling interval
            country_iso_code='CHE',
            log_level='warning',
        )
        tracker.start()

        try:
            # track time
            train_start = time.perf_counter()

            # set seed for each run
            set_seed_locally(62 + init * 1000)
            set_seed(62 + init * 1000)

            # init new model
            if lora:
                model = model_init(
                    model_ckpt,
                    device,
                    lora=True,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    lora_target_modules=lora_target_modules,
                    lora_bias=lora_bias,
                    lora_modules_to_save=lora_modules_to_save,
                )
            else:
                model = model_init(model_ckpt, device)

            # Initialize the data collator
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            # set up training arguments
            training_args = TrainingArguments(
                output_dir=f'{output_path}/init_{init}/{model_tag}',
                evaluation_strategy='epoch',
                save_strategy='epoch',
                save_total_limit=1,  # limit to save only one checkpoint to manage disk space usage
                learning_rate=learning_rate,
                num_train_epochs=num_epochs,
                warmup_ratio=warmup_ratio,
                warmup_steps=warmup_steps,
                weight_decay=weight_decay,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=4,
                gradient_checkpointing=not lora,
                disable_tqdm=False,
                log_level='info',
                logging_dir=f'{output_path}/init_{init}/logs',
                logging_steps=500,
                fp16 = device.type == 'cuda',   # fp16 only possible on GPU, otherwise raises ValueError
                load_best_model_at_end=True,
                metric_for_best_model='accuracy',
                report_to=[],
            )
            # Initialize the Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                compute_metrics=compute_metrics,
                train_dataset=tokenized_hf_train,
                eval_dataset=tokenized_hf_val,
                data_collator=data_collator,
                tokenizer=tokenizer
            )
            # train the model
            last_checkpoint = None
            if os.path.isdir(training_args.output_dir):
                last_checkpoint = get_last_checkpoint(training_args.output_dir)

            if last_checkpoint is not None:
                print(f'Resuming training from checkpoint: {last_checkpoint}')
                trainer.train(resume_from_checkpoint=last_checkpoint)
            else:
                print('No valid checkpoint found. Training from scratch.')
                trainer.train()
            print(trainer.state.log_history)
            # Finish time tracking
            train_end = time.perf_counter()
            train_time_seconds = train_end - train_start
            train_time_hours = train_time_seconds / 3600.0
            # save logs
            history = pd.DataFrame(trainer.state.log_history)
            history.to_csv(f'{output_path}/init_{init}/log_history.csv')
            empty_cache()
            trainer.model.save_pretrained(f'{output_path}/init_{init}/{model_tag}-final')
            # prediction and result saving
            preds_output = trainer.predict(tokenized_hf_test)
            with open(f'{output_path}/init_{init}/results.csv', 'w') as f:
                for key, value in preds_output.metrics.items():
                    f.write(f'{key},{value}\n')
                f.write('Parameters used:\n')
                f.write(f'learning_rate = {learning_rate}\n')
                f.write(f'num_epochs = {num_epochs}\n')
                f.write(f'batch_size = {batch_size}\n')
                f.write(f'weight_decay = {weight_decay}\n')
                f.write(f'warmup_steps = {warmup_steps}\n')
                f.write(f'warmup_ratio = {warmup_ratio}\n')
                if lora:
                    f.write(f'lora = {lora}\n')
                    f.write(f'lora_r = {lora_r}\n')
                    f.write(f'lora_alpha = {lora_alpha}\n')
                    f.write(f'lora_dropout = {lora_dropout}\n')
                    f.write(f'lora_target_modules = {lora_target_modules}\n')
                    f.write(f'lora_bias = {lora_bias}\n')
                    f.write(f'lora_modules_to_save = {lora_modules_to_save}\n')
                f.write(f'training set: {len(tokenized_hf_train)} samples\n')
                f.write(f'validation set: {len(tokenized_hf_val)} samples\n')
                f.write(f'test set: {len(tokenized_hf_test)} samples\n')
                f.write(f'train_time_seconds = {train_time_seconds}\n')
                f.write(f'train_time_hours = {train_time_hours}\n')
                f.write(f'Seed used: {62 + init * 1000}')
            y_pred = np.argmax(preds_output.predictions, axis=1)
            np.save(f'{output_path}/init_{init}/y_pred.npy', y_pred)
            accuracy = preds_output.metrics['test_accuracy']
            # build and save confusion matrix
            build_confusion_matrix(tokenized_hf_test, y_pred, f'{output_path}/init_{init}')
            initializations_accuracies[f'init_{init}'] = accuracy
            print(f'Initialization {init + 1} with accuracy={accuracy}')

        finally:
            try:
                emissions_kg = tracker.stop()
                with open(f'{init_base_dir}/emissions_summary.txt', 'w') as f:
                    f.write(f'emissions_kg={emissions_kg}\n')
            except Exception as carbon_error:
                print(f'CodeCarbon failed while stopping tracker: {carbon_error}')

    with open(f'{output_path}/initialization_results.csv', 'w') as f:
        f.write(f'Accuracies for each initialization: {json.dumps(initializations_accuracies, indent=4)}')