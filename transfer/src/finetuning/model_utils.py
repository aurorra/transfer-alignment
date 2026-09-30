from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model


def model_init(
        model_ckpt,
        device,
        output_hidden_states=False,
        lora=False,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules=None,
        lora_bias='none',
        lora_modules_to_save=None,
):
    m = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt,
        num_labels=2,
        output_hidden_states=output_hidden_states
    ).to(device)

    if not lora:
        return m

    if lora_target_modules is None:
        lora_target_modules = ['query', 'value']

    if lora_modules_to_save is None:
        lora_modules_to_save = ['classifier']

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias=lora_bias,
        modules_to_save=lora_modules_to_save,
    )

    m = get_peft_model(m, lora_config)
    return m.to(device)