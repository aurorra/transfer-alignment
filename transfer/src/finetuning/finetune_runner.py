from pathlib import Path

from transformers import AutoTokenizer
import torch

from transfer.src.finetuning.trainer_utils import empty_cache, set_seed_locally, set_seed, training_loop
from transfer.src.data.data_utils import load_data, convert_to_hf, tokenize_data


def run_finetuning(config: dict):
    required = ['model_ckpt', 'path_train', 'path_val', 'path_test', 'output_path']
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Missing config fields: {', '.join(missing)}")

    # Model config
    model_ckpt = config['model_ckpt']
    tok = config.get('tokenizer', model_ckpt)
    learning_rate = config.get('learning_rate', 5e-5)
    batch_size = config.get('batch_size', 32)
    num_epochs = config.get('num_epochs', 3)
    weight_decay = config.get('weight_decay', 0.0)
    warmup_steps = config.get('warmup_steps', 0)
    warmup_ratio = config.get('warmup_ratio', 0)
    num_inits = config.get('num_inits', 1)
    lora = config.get('lora', False)

    # Data paths
    path_train = config['path_train']
    path_val = config['path_val']
    path_test = config['path_test']
    additional_samples_path = config.get('additional_samples_path', None)

    # Output path
    output_path = Path(config['output_path'])

    # GPU or CPU usage, raise error if GPU is not accessible but expected
    require_cuda = config.get('require_cuda', True)
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA is not available. This finetuning job is expected to run on a GPU. '
            'Check your Slurm script, partition, --gres setting, and PyTorch/CUDA environment.'
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f'Using CUDA device: {torch.cuda.get_device_name(0)}')
    else:
        print('Using CPU.')

    # Ensure reproducibility
    empty_cache()
    set_seed_locally(62)
    set_seed(62)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tok)

    # Check if files exist
    for key in ['path_train', 'path_val', 'path_test']:
        path = Path(config[key])
        if not path.is_file():
            raise FileNotFoundError(f'{key} does not exist: {path.resolve()}')

    # Load & prepare data
    df_train, df_val, df_test = load_data(path_train, path_val, path_test, additional_samples_path)
    hf_train, hf_val, hf_test = convert_to_hf(df_train, df_val, df_test)
    tokenized_train, tokenized_val, tokenized_test = tokenize_data(hf_train, hf_val, hf_test, tokenizer)

    output_path.mkdir(parents=True, exist_ok=True)
    print(f'Using output directory: {output_path.resolve()}')

    # Run training
    training_kwargs = dict(
        model_ckpt=model_ckpt,
        tokenizer=tokenizer,
        tokenized_hf_train=tokenized_train,
        tokenized_hf_val=tokenized_val,
        tokenized_hf_test=tokenized_test,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        output_path=output_path,
        num_inits=num_inits,
        device=device,
    )

    if lora:
        training_kwargs.update(
            lora=True,
            lora_r=config.get('lora_r', 4),
            lora_alpha=config.get('lora_alpha', 8),
            lora_dropout=config.get('lora_dropout', 0.0),
            lora_target_modules=config.get('lora_target_modules', ['query', 'value']),
            lora_bias=config.get('lora_bias', 'none'),
            lora_modules_to_save=config.get('lora_modules_to_save', ['classifier']),
        )

    training_loop(**training_kwargs)
