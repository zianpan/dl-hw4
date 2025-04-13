# %% [markdown]
# # Imports
# 
# - If your setup was done correctly, you should be able to run the following cell without any issues.

# %%
from hw4lib.data import (
    H4Tokenizer,
    LMDataset,
    verify_dataloader
)
from hw4lib.model import (
    CausalMask,
    PadMask,
    PositionalEncoding,
    DecoderOnlyTransformer
)
from hw4lib.utils import (
    create_optimizer,
    create_scheduler,
    plot_lr_schedule
)
from hw4lib.trainers import (
    LMTrainer,
)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import yaml
import gc
import torch
from torchinfo import summary
import os
import json
import tarfile
import shutil
import wandb
import yaml
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "mps"
print(f"Using device: {device}")

# %% [markdown]
# # Experiments
# From this point onwards you may want to switch to a `GPU` runtime. 
# - `OBJECTIVE`: You must achieve a per-character perplexity ≤ 3.5 in order to get points for Task 2.

# %% [markdown]
# ## Config
# - You can use the `config.yaml` file to set your config for your ablation study.
# 
# ---
# ### Notes:
# 
# - Set `tokenization: token_type:` to specify your desired tokenization strategy
# - You will need to set the root path to your `hw4p1_data` folder in `data: root:`. This will depend on your setup. For eg. if you are following out setup instruction:
#   - `PSC`: `"/local/hw4_data/hw4p1_data"`
#   - `Colab:`: `"/content/hw4_data/hw4p1_data"`
#   - `Kaggle:`: `"/kaggle/input/s25-hw4-data/hw4p1_data"`
# - There's extra configurations in the `optimizer` section which will only be relevant if you decide to use the `create_optimizer` function we've provided in `hw4lib/utils/create_optimizer.py`.
# - `BE CAREFUL` while setting numeric values. Eg. `1e-4` will get serialized to a `str` while `1.0e-4` gets serialized to float. 
# 



# %%
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# %%
run_name = config['training']['run_name']

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
train_dataset  = LMDataset(
    partition  = config['data']['train_partition'],
    config     = config['data'],
    tokenizer  = Tokenizer
)

val_dataset    = LMDataset(
    partition  = config['data']['val_partition'],
    config     = config['data'],
    tokenizer  = Tokenizer
)

test_dataset   = LMDataset(
    partition  = config['data']['test_partition'],
    config     = config['data'],
    tokenizer  = Tokenizer
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

# %% [markdown]
# ### Dataloader Verification

# %%
verify_dataloader(train_loader)

# %%
verify_dataloader(val_loader)

# %%
verify_dataloader(test_loader)

# %% [markdown]
# ## Calculate Max Transcript Length
# 
# 
# 

# %% [markdown]
# Calculating the maximum transcript length across your dataset is a crucial step when working with certain transformer models.
# -  We'll use sinusoidal positional encodings that must be precomputed up to a fixed maximum length.
# - This maximum length is a hyperparameter that determines:
#   - How long of a sequence your model can process
#   - The size of your positional encoding matrix
#   - Memory requirements during training and inference
# - `Requirements`: For this assignment, ensure your positional encodings can accommodate at least the longest sequence in your dataset to prevent truncation. However, you can set this value higher if you anticipate using your language model to work with longer sequences in future tasks (hint: this might be useful for P2! 😉).

# %%
max_transcript_length = max(train_dataset.text_max_len, val_dataset.text_max_len, test_dataset.text_max_len)
print("="*50)
print(f"{'Global Max Transcript Length':<30} : {max_transcript_length}")
print("="*50)

# %% [markdown]
# ## Model

# %%
model_config = config['model']
model_config.update({
    'max_len': max_transcript_length,
    'num_classes': Tokenizer.vocab_size
})
model = DecoderOnlyTransformer(**model_config)

# Get some inputs from the text loader
for batch in train_loader:
    shifted_transcripts, golden_transcripts, transcript_lengths = batch
    print("Shape of shifted_transcripts : ", shifted_transcripts.shape)
    print("Shape of golden_transcripts  : ", golden_transcripts.shape)
    print("Shape of transcript_lengths  : ", transcript_lengths.shape)
    break

model_stats = summary(model, input_data=[shifted_transcripts, transcript_lengths])
print(model_stats)

# %% [markdown]
# ## Wandb

# %%
wandb.login(key="8475199febe13b3465c7d5e4a595bba7422c14fc")

# %% [markdown]
# ## Trainer
# 
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

# %%
model = model.to(device)

# %%
trainer = LMTrainer(
    model=model,
    tokenizer=Tokenizer,
    config=config,
    run_name=run_name,
    config_file="config.yaml",
    device=device
)

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

# %% [markdown]
# #### Creating a test scheduler and plotting the learning rate schedule

# %%
test_scheduler = create_scheduler(
    optimizer=trainer.optimizer,
    scheduler_config=config['scheduler'],
    train_loader=train_loader,
    gradient_accumulation_steps=config['training']['gradient_accumulation_steps']
)

plot_lr_schedule(
    scheduler=test_scheduler,
    num_epochs=config['training']['epochs'],
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
# # Train
# - Set your epochs

# %%
trainer.train(train_loader, val_loader, epochs=config['training']['epochs'])

# %% [markdown]
# # Evaluate
# 

# %%
test_metrics, test_generation_results = trainer.evaluate(test_loader)
# Cleanup
trainer.cleanup()

# %% [markdown]
# # Submission
# To submit your assignment, you will need to create a `handin.tar` with the following directory structure:
# 
# ```
# handin/
# ├── mytorch/                     # Your implemented modules
# ├── test_metrics.json            # Results from evaluation
# ├── test_generated_results.json  # Sample text generations
# └── model_arch.txt               # Model architecture summary
# ```
# 
# - Simply run the cell below once you are satisfied with your current state and this will create the `handin.tar` file.
# - After running the above cell, you should see the handin.tar file in the current directory
# - Upload the `handin.tar` file to the `HW4P1` assignment on Autolab.

# %%
# Create temporary handin directory
if os.path.exists('handin'):
    shutil.rmtree('handin')
os.makedirs('handin')

# Copy mytorch directory
shutil.copytree('mytorch', 'handin/mytorch')

# Save final results
with open('handin/test_metrics.json', 'w') as f:
    json.dump(test_metrics, f, indent=4)

with open('handin/test_generated_results.json', 'w') as f:
    json.dump(test_generation_results['greedy'], f, indent=4)

# Save model architecture
with open('handin/model_arch.txt', 'w') as f:
    f.write(str(model_stats))

# Create tar file with all exclusions handled by filter
with tarfile.open('handin.tar', 'w') as tar:
    def filter_files(tarinfo):
        # Skip unwanted files
        if any(pattern in tarinfo.name for pattern in [
            '.DS_Store',
            '__pycache__',
            '.pyc'
        ]):
            return None
        return tarinfo

    tar.add('handin', arcname='handin', filter=filter_files)

# Cleanup
shutil.rmtree('handin')

print("Created handin.tar successfully!")

## After running the above cell, you should see the handin.tar file in the current directory

# %%



