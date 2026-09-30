import os
import random
import pickle

import pandas as pd
from datasets import Dataset

REQUIRED_COLUMNS_RAW = {'text', 'target', 'label'}


def combine_dataset(paths: list[str], type_names: list[str], output_path):
    dfs = []
    for i in range(len(paths)):
        df = pd.read_csv(paths[i])
        df['error_type'] = type_names[i]

        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(keep='first')
    combined = combined.sample(frac=1.0, random_state=64).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    combined.to_csv(output_path, index=False)


def validate_raw_dataframe(df: pd.DataFrame, split_name: str) -> None:
    missing = REQUIRED_COLUMNS_RAW - set(df.columns)
    if missing:
        raise ValueError(
            f'{split_name} is missing required columns: {sorted(missing)}. '
            f'Available columns: {list(df.columns)}'
        )

    if df.empty:
        raise ValueError(f'{split_name} is empty.')

    invalid_labels = sorted(set(df['label'].dropna().unique()) - {0, 1})
    if invalid_labels:
        raise ValueError(
            f'{split_name} contains labels other than 0/1: {invalid_labels}'
        )

    empty_sentence_mask = df['text'].astype(str).str.strip().eq('')
    if empty_sentence_mask.any():
        n_empty = int(empty_sentence_mask.sum())
        raise ValueError(
            f'{split_name} contains {n_empty} empty sentences. '
            'Remove them or decide explicitly how to handle them.'
        )

    if df['label'].isna().any():
        raise ValueError(f'{split_name} contains missing label values.')


def load_data(path_train, path_val, path_test, additional_samples_path=None):
    """
    Load data and validate entry correctness. Add additional samples to training data and shuffle dataset if path given.

    :param path_train: CSV file containing training data
    :param path_val: CSV file containing validation data
    :param path_test: CSV file containing test data
    :param additional_samples_path:
    :return train_dataset: training set as pandas DataFrame, val_dataset: evaluation set as pandas DataFrame,
                test_dataset: test set as pandas DataFrame
    """
    # Load data from CSV files
    train_dataset = pd.read_csv(path_train, keep_default_na=False).rename(
        columns={'sentence': 'text', 'correction': 'target'})
    val_dataset = pd.read_csv(path_val, keep_default_na=False).rename(
        columns={'sentence': 'text', 'correction': 'target'})
    test_dataset = pd.read_csv(path_test, keep_default_na=False).rename(
        columns={'sentence': 'text', 'correction': 'target'})

    # Validate data entries
    validate_raw_dataframe(train_dataset, 'training')
    validate_raw_dataframe(val_dataset, 'validation')
    validate_raw_dataframe(test_dataset, 'test')

    # Add additional training sample to training data and shuffle dataset
    if additional_samples_path is not None:
        additional_samples = pd.read_csv(additional_samples_path, keep_default_na=False).rename(
            columns={'sentence': 'text', 'correction': 'target'})

        validate_raw_dataframe(additional_samples, 'additional samples')

        train_dataset = pd.concat([train_dataset, additional_samples], ignore_index=True)
        train_dataset = train_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f'Added {additional_samples.shape[0]} additional samples from {additional_samples_path}')

    return train_dataset, val_dataset, test_dataset


def convert_to_hf(train_dataset, val_dataset, test_dataset):
    """
    Convert pandas DataFrame to a HuggingFace Dataset with columns 'text' and 'label'.

    :param train_dataset: training set as pandas DataFrame
    :param val_dataset: evaluation set as pandas DataFrame
    :param test_dataset: test set as pandas DataFrame
    :return: hf_train_dataset: training set as a HF Dataset, hf_val_dataset: validation dataset as a HF Dataset,
                hf_test_dataset: test dataset as a HF Dataset
    """
    required = {'text', 'label'}

    for name, df in [
        ('training', train_dataset),
        ('validation', val_dataset),
        ('test', test_dataset),
    ]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f'{name} dataframe is missing columns after preprocessing: {sorted(missing)}. '
                f'Available columns: {list(df.columns)}'
            )

    hf_train_dataset = Dataset.from_pandas(train_dataset[['text', 'label']], preserve_index=False)
    hf_val_dataset = Dataset.from_pandas(val_dataset[['text', 'label']], preserve_index=False)
    hf_test_dataset = Dataset.from_pandas(test_dataset[['text', 'label']], preserve_index=False)

    return hf_train_dataset, hf_val_dataset, hf_test_dataset


def tokenize_data(hf_train_dataset, hf_val_dataset, hf_test_dataset, tokenizer):
    """
    Tokenizes data with map function and sets it into torch format.
    """
    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt')

    # Tokenize in batches of 1000 examples
    tokenized_hf_train_dataset = hf_train_dataset.map(tokenize, batched=True, batch_size=1000)
    tokenized_hf_val_dataset = hf_val_dataset.map(tokenize, batched=True, batch_size=1000)
    tokenized_hf_test_dataset = hf_test_dataset.map(tokenize, batched=True, batch_size=1000)

    # Model expects tensors as inputs -> convert the input_ids and attention_mask columns to the 'torch' format
    tokenized_hf_train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenized_hf_val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenized_hf_test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    return tokenized_hf_train_dataset, tokenized_hf_val_dataset, tokenized_hf_test_dataset


def sample_additional_sentences(pickle_path: str, n: int, output_path: str, seed: int = 42):
    """ Loads data from pickle file, randomly samples n samples, and saves them into a CSV file. """
    random.seed(seed)

    # Load data from pickle file
    with open(pickle_path, 'rb') as handle:
        data = pickle.load(handle)

    # Randomly select n samples, change sharp s to ss for Swiss writing standard
    random_samples = random.sample(data, n)
    random_samples = [{
        **s,  # keep all original keys
        'sentence': s['sentence'].replace('ß', 'ss'),
        'correction': s['correction'].replace('ß', 'ss')
    } for s in random_samples]

    df = pd.DataFrame(random_samples)

    os.makedirs(output_path, exist_ok=True)

    df.to_csv(f'{output_path}/added_samples_n={n}.csv', index=False)


def sample_additional_sentences_stratified_2level(data_path: str, n: int, output_path: str, seed: int = 42):
    """
    Loads data from a CSV file, randomly samples n stratified (label and error type) samples,
    and saves them into a CSV file.
    """
    random.seed(seed)

    df = pd.read_csv(data_path)
    error_types = set(df['error_type'].to_list())

    if n / 2 / len(error_types) % 1 != 0:
        print(f'n must be chosen as n / 2 / len(error_types) % 1 == 0')
        exit(-1)

    selections = []
    for error_type in error_types:
        df_type = df[df['error_type'] == error_type]
        # Split by class
        df_type_0 = df_type[df_type['label'] == 0]
        df_type_1 = df_type[df_type['label'] == 1]
        need_per_class = int(n / 2 / len(error_types))

        # Sample an equal number from each label within each error type
        samp_type_0 = df_type_0.sample(n=need_per_class, random_state=seed)
        samp_type_1 = df_type_1.sample(n=need_per_class, random_state=seed)

        # Concatenate and shuffle to interleave labels
        sel_type = pd.concat([samp_type_0, samp_type_1]).sample(frac=1.0, random_state=seed)
        selections.append(sel_type)

    sel = pd.concat(selections).sample(frac=1.0, random_state=seed).reset_index()
    sel = sel.drop(['index'], axis=1)

    os.makedirs(output_path, exist_ok=True)

    sel.to_csv(f'{output_path}/added_samples_n={n}_stratified.csv', index=False)


def sample_sentences_stratified_label(data_path: str, n: int, output_path: str, seed: int = 42) -> None:
    """
    Loads data from a CSV file, randomly samples n stratified samples total (balanced across labels 0 and 1),
    then splits into stratified train/val with 90% train and 10% val, and saves to CSV files.
    Expects a column named 'label' with values 0/1.
    """
    # Load data
    df = pd.read_csv(data_path)

    if 'label' not in df.columns:
        raise ValueError('Input CSV must contain a \'label\' column.')

    # Number of samples per class and training proportion
    need_per_class = n // 2
    train_frac = 0.9
    n_train = int(need_per_class * train_frac)

    # Split by class
    df_type_0 = df[df['label'] == 0]
    df_type_1 = df[df['label'] == 1]

    if n % 2 != 0:
        raise ValueError(f'n must be even for balanced binary sampling, got n={n}.')

    if len(df_type_0) < need_per_class:
        raise ValueError(
            f'Not enough label=0 samples. Need {need_per_class}, found {len(df_type_0)}.'
        )

    if len(df_type_1) < need_per_class:
        raise ValueError(
            f'Not enough label=1 samples. Need {need_per_class}, found {len(df_type_1)}.'
        )

    # Shuffle once per class and take exactly need_per_class
    df_type_0 = df_type_0.sample(frac=1.0, random_state=seed).reset_index(drop=True).iloc[:need_per_class]
    df_type_1 = df_type_1.sample(frac=1.0, random_state=seed).reset_index(drop=True).iloc[:need_per_class]

    # Split directly (disjoint)
    train_0, val_0 = df_type_0.iloc[:n_train], df_type_0.iloc[n_train:]
    train_1, val_1 = df_type_1.iloc[:n_train], df_type_1.iloc[n_train:]

    # Merge and shuffle
    train_df = pd.concat([train_0, train_1]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat([val_0, val_1]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    os.makedirs(output_path, exist_ok=True)

    train_df.to_csv(f'{output_path}/training.csv', index=False)
    val_df.to_csv(f'{output_path}/validation.csv', index=False)
