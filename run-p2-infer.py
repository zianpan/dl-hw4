# %%
from hw4lib.data import (
    H4Tokenizer,
    ASRDataset,
    verify_dataloader
)
from hw4lib.model import (
    DecoderOnlyTransformer,
    EncoderDecoderTransformer
)
from hw4lib.utils import (
    create_scheduler,
    create_optimizer,
    plot_lr_schedule
)
from hw4lib.trainers import (
    ASRTrainer,
    ProgressiveTrainer
)
from torch.utils.data import DataLoader
import yaml
import gc
import torch
from torchinfo import summary
import os
import json
import wandb
import pandas as pd
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# %%
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# %% [markdown]
# ## Tokenizer

# %%
Tokenizer = H4Tokenizer(
    token_map  = config['tokenization']['token_map'],
    token_type = config['tokenization']['token_type']
)

# %% [markdown]
# ## Datasets

# %%
train_dataset = ASRDataset(
    partition=config['data']['train_partition'],
    config=config['data'],
    tokenizer=Tokenizer,
    isTrainPartition=True,
    global_stats=None  # Will compute stats from training data
)

# TODO: Get the computed global stats from training set
global_stats = None
if config['data']['norm'] == 'global_mvn':
    global_stats = (train_dataset.global_mean, train_dataset.global_std)
    print(f"Global stats computed from training set.")

val_dataset = ASRDataset(
    partition=config['data']['val_partition'],
    config=config['data'],
    tokenizer=Tokenizer,
    isTrainPartition=False,
    global_stats=global_stats
)

test_dataset = ASRDataset(
    partition=config['data']['test_partition'],
    config=config['data'],
    tokenizer=Tokenizer,
    isTrainPartition=False,
    global_stats=global_stats
)

gc.collect()

# %% [markdown]
# ## Dataloaders

# %%
train_loader    = DataLoader(
    dataset     = train_dataset,
    batch_size  = config['data']['batch_size'],
    shuffle     = True,
    num_workers = config['data']['NUM_WORKERS'] if device == 'cuda' else 0,
    pin_memory  = True,
    collate_fn  = train_dataset.collate_fn
)

val_loader      = DataLoader(
    dataset     = val_dataset,
    batch_size  = config['data']['batch_size'],
    shuffle     = False,
    num_workers = config['data']['NUM_WORKERS'] if device == 'cuda' else 0,
    pin_memory  = True,
    collate_fn  = val_dataset.collate_fn
)

test_loader     = DataLoader(
    dataset     = test_dataset,
    batch_size  = config['data']['batch_size'],
    shuffle     = False,
    num_workers = config['data']['NUM_WORKERS'] if device == 'cuda' else 0,
    pin_memory  = True,
    collate_fn  = test_dataset.collate_fn
)

gc.collect()

# %% [markdown]
# ## Calculate Max Lengths
# Calculating the maximum transcript length across your dataset is a crucial step when working with certain transformer models.
# -  We'll use sinusoidal positional encodings that must be precomputed up to a fixed maximum length.
# - This maximum length is a hyperparameter that determines:
#   - How long of a sequence your model can process
#   - The size of your positional encoding matrix
#   - Memory requirements during training and inference
# - `Requirements`: For this assignment, ensure your positional encodings can accommodate at least the longest sequence in your dataset to prevent truncation. However, you can set this value higher if you anticipate using your languagemodel to work with longer sequences in future tasks (hint: this might be useful for P2! 😉).
# - `NOTE`: We'll be using the same positional encoding matrix for all sequences in your dataset. Take this into account when setting your maximum length.

# %%
max_feat_len       = max(train_dataset.feat_max_len, val_dataset.feat_max_len, test_dataset.feat_max_len)
max_transcript_len = max(train_dataset.text_max_len, val_dataset.text_max_len, test_dataset.text_max_len)
max_len            = max(max_feat_len, max_transcript_len)

print("="*50)
print(f"{'Max Feature Length':<30} : {max_feat_len}")
print(f"{'Max Transcript Length':<30} : {max_transcript_len}")
print(f"{'Overall Max Length':<30} : {max_len}")
print("="*50)

# %% [markdown]
# ## Wandb

# %%
wandb.login(key="8475199febe13b3465c7d5e4a595bba7422c14fc")

# %% [markdown]
# ## Training
# Every time you run the trainer, it will create a new directory in the `expts` folder with the following structure:
# ```
# expts/
#     └── {run_name}/
#         ├── config.yaml
#         ├── model_arch.txt
#         ├── checkpoints/
#         │   ├── checkpoint-best-metric-model.pth
#         │   └── checkpoint-last-epoch-model.pth
#         ├── attn/
#         │   └── {attention visualizations}
#         └── text/
#             └── {generated text outputs}
# ```
# 

# %% [markdown]
# ### Training Strategy 1: Cold-Start Trainer

# %% [markdown]
# #### Model Load (Default)

# %%
model_config = config['model'].copy()
model_config.update({
    'max_len': max_len,
    'num_classes': Tokenizer.vocab_size
})

model = EncoderDecoderTransformer(**model_config)

# Get some inputs from the train dataloader
for batch in train_loader:
    padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths = batch
    break


model_stats = summary(model, input_data=[padded_feats, padded_shifted, feat_lengths, transcript_lengths])
print(model_stats)

# %% [markdown]
# #### Initialize Trainer
# 
# If you need to reload the model from a checkpoint, you can do so by calling the `load_checkpoint` method.
# 
# ```python
# checkpoint_path = "path/to/checkpoint.pth"
# trainer.load_checkpoint(checkpoint_path)
# ```
# 

# %%
trainer = ASRTrainer(
    model=model,
    tokenizer=Tokenizer,
    config=config,
    run_name="p2-complex-t1",
    config_file="config.yaml",
    device=device
)

checkpoint_path = "/root/hw4/dl-hw4/expts/deeper_transformer_specaug/checkpoints/checkpoint-best-metric-model.pth"
trainer.load_checkpoint(checkpoint_path)

# %% [markdown]
# ### Setup Optimizer and Scheduler
# 
# You can set your own optimizer and scheduler by setting the class members in the `LMTrainer` class.
# Eg:
# ```python
# trainer.optimizer = optim.AdamW(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
# trainer.scheduler = optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=config['training']['epochs'])
# ```
# 
# We also provide a utility function to create your own optimizer and scheduler with the congig and some extra bells and whistles. You are free to use it or not. Do read their code and documentation to understand how it works (`hw4lib/utils/*`).
# 

# %% [markdown]
# #### Setting up the optimizer

# %%
trainer.optimizer = create_optimizer(
    model=model,
    opt_config=config['optimizer']
)

# trainer.optimizer = optim.AdamW(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
# trainer.scheduler = optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=config['training']['epochs'])
# %% [markdown]
# #### Creating a test scheduler and plotting the learning rate schedule

# %%
test_scheduler = create_scheduler(
    optimizer=trainer.optimizer,
    scheduler_config=config['scheduler'],
    train_loader=train_loader,
    gradient_accumulation_steps=config['training']['gradient_accumulation_steps']
)


# %% [markdown]
# #### Setting up the scheduler

# %%
trainer.scheduler = create_scheduler(
    optimizer=trainer.optimizer,
    scheduler_config=config['scheduler'],
    train_loader=train_loader,
    gradient_accumulation_steps=config['training']['gradient_accumulation_steps']
)

# %% [markdown]
# #### Train
# - Set your epochs
torch.cuda.empty_cache()
import gc
gc.collect()
# %%
# trainer.train(train_loader, val_loader, epochs=config['training']["epochs"])

# %% [markdown]
# #### Inference
# 

# %%
# Define the recognition config: Greedy search
recognition_config = {
    'num_batches': None,
    'temperature': 1.0,
    'repeat_penalty': 1.0,
    'lm_weight': None,
    'lm_model': None,
    'beam_width': 6, # Beam width of 1 reverts to greedy
}

# Recognize with the shallow fusion config
config_name = "test"
print(f"Evaluating with {config_name} config")
results = trainer.recognize(test_loader, recognition_config, config_name=config_name, max_length=max_transcript_len)


# Calculate metrics on full batch
generated = [r['generated'] for r in results]
results_df = pd.DataFrame(
    {
        'id': range(len(generated)),
        'transcription': generated
    }
)

# Cleanup (Will end wandb run)
trainer.cleanup()

# %% [markdown]
# ## Submit to Kaggle

# %% [markdown]
# ### Authenticate Kaggle
# In order to use the Kaggle’s public API, you must first authenticate using an API token. Go to the 'Account' tab of your user profile and select 'Create New Token'. This will trigger the download of kaggle.json, a file containing your API credentials.
# - `TODO`: Set your kaggle username and api key here based on the API credentials listed in the kaggle.json
# 
# 
# 

# %%
import os
os.environ["KAGGLE_USERNAME"] = "your_kaggle_username_here"
os.environ["KAGGLE_KEY"] = "your_kaggle_api_key_here"

# %%
results_df.head()

# %% [markdown]
# ### Submit

# %%
results_df.to_csv("results.csv", index=False)
# !kaggle competitions submit -c 11785-s25-hw4p2-asr -f results.csv -m "My Submission"

# %%



