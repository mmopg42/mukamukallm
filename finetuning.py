from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

import argparse

######################
######## 변수 ########
######################

data_directory = 'mukamukallm/data'
output_dir = 'mukamukallm/model'


# 데이터를 포맷팅합니다
def formatting_data(input):
    prompt_format = """상황: 
{}

대화 기록: 
{}"""

    backgrounds = input['background']
    dialogues = input['dialogue']

    texts = []
    for background, dialogue in zip(backgrounds, dialogues):
        dialogue = dialogue.replace("\n\n", "<eos>")
        text = prompt_format.format(background, dialogue)
        texts.append(text)

    return {"text": texts}


# 학습 코드드
def train():
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = 'unsloth/gemma-2-9b',
        max_seq_length = 2048,
        dtype=None,
        load_in_4bit=True
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules= [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            ],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = 'none',
        use_gradient_checkpointing='unsloth',
        random_state = 5252,
        use_rslora = False,
        loftq_config=None
    )

    dataset = load_dataset(data_directory)


    dataset_mapping = dataset.map(formatting_data, batched=True)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset_mapping['train'],
        dataset_text_field = "text",
        max_seq_length = 2048,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs=30,
            learning_rate = 1e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 5252,
            output_dir = output_dir,
            report_to = "none", # Use this for WandB etc
        ),
    )

    trainer_stats = trainer.train()


if __name__ == "__main__":
    train()
