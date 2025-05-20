import torch
import torch.nn as nn
import wandb

from model import build_transformer
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from pathlib import Path
from dataset import BilingualDataset, causal_mask
from torch.utils.tensorboard import SummaryWriter
from config import get_weights_file_path, get_config, get_latest_weights_file_path
from tqdm import tqdm
import os
import argparse
import torchmetrics

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

def greedy_decode(model, encoder_input, encoder_mask, src_tokenizer, target_tokenizer, max_len, device):
    sos_idx = target_tokenizer.token_to_id('[SOS]')
    eos_idx = target_tokenizer.token_to_id('[EOS]')
    print("SOS Index:", sos_idx)
    print("EOS Index:", eos_idx)

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.module.encode(encoder_input, encoder_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(device)
    model.eval()
    with torch.no_grad():
        while True:
            if decoder_input.size(1) == max_len:
                break

            # build mask for target
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)

            # calculate output
            out = model.module.decode(encoder_output, decoder_input, encoder_mask, decoder_mask)

            # get next token
            prob = model.module.project(out[:, -1])
            # print(f"Probabilities: {prob}")
            topk = prob.topk(5)
            _, next_word = torch.max(prob, dim=1)
            # print(f"Next word ID: {next_word.item()}")
            print("Top tokens:", target_tokenizer.decode(topk.indices[0].tolist()))
            print("Probs:", topk.values.tolist())
            token = target_tokenizer.id_to_token(next_word.item())
            print(f"Predicted ID: {next_word.item()} -> '{token}'")
            print("Decoder_input: ", decoder_input)
            print(f"Encoder_mask-shape: {encoder_mask.shape} Decoder_mask-shape: {decoder_mask.shape}")
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(encoder_input).fill_(next_word.item()).to(device)], dim=1
            )

            if next_word == eos_idx:
                break

    return decoder_input.squeeze(0)


def run_validation(model, src_tokenizer, target_tokenizer, writer, global_step, validation_ds, print_msg, device, max_len, num_examples=2):
    model.module.eval()
    count = 0
    console_width = 80
    source_texts = []
    expected = []
    predicted = []
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            assert encoder_input.shape[0] == 1, "Batch size must be 1"
            model_out = greedy_decode(model, encoder_input, encoder_mask, src_tokenizer, target_tokenizer, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["target_text"][0]
            model_output_text = target_tokenizer.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_output_text)

            #Print to the console
            print_msg("-"*console_width)
            print_msg(f"SOURCE:{source_text}")
            print_msg(f"TARGET:{target_text}")
            print_msg(f"MODEL_OUTPUT:{model_output_text}")

            if count == num_examples:
                break
        
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        wandb.log({'validation/cer': cer, 'global_step': global_step})
        
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        wandb.log({'validation/wer': wer, 'global_step': global_step})

        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        wandb.log({'validation/BLEU': bleu, 'global_step': global_step})


def get_all_sentences(ds, language):
    for item in ds:
        yield item["translation"][language]

def get_or_build_tokenizer(config, ds, language):
    tokenizer_path = Path(config["tokenizer_path"].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, language), trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset("atrisaxena/mini-iitb-english-hindi", split="train")

    # Build the tokenizers
    src_tokenizer = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    target_tokenizer = get_or_build_tokenizer(config, ds_raw, config["lang_target"])
    print("Target vocab size:", target_tokenizer.get_vocab_size())

    # Split the dataset
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = int(len(ds_raw)) - int(train_ds_size)
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, src_tokenizer, target_tokenizer, config["lang_src"], config["lang_target"], config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw, src_tokenizer, target_tokenizer, config["lang_src"], config["lang_target"], config["seq_len"])

    max_src_len = 0
    max_target_len = 0

    for item in ds_raw:
        src_input = src_tokenizer.encode(item["translation"][config["lang_src"]]).ids
        target_input = target_tokenizer.encode(item["translation"][config["lang_target"]]).ids
        max_src_len = max(max_src_len, len(src_input))
        max_target_len = max(max_target_len, len(target_input))
    
    print(f"Maximum length of src inputs: {max_src_len}")
    print(f"Maximum length of target inputs: {max_target_len}")

    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=False, sampler=DistributedSampler(train_ds, shuffle=True))
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, src_tokenizer, target_tokenizer

def get_model(config, src_vocab_size, target_vocab_size):
    train_transformer = build_transformer(src_vocab_size, target_vocab_size, config["seq_len"], config["seq_len"], d_model=config["d_model"])
    return train_transformer    


def train_model(config):
    assert torch.cuda.is_available(), "Training on CPU not supported"
    local_rank = config["local_rank"]
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    print(f"GPU: {config['local_rank']} - Using device: {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, src_tokenizer, target_tokenizer = get_ds(config)

    model = get_model(config, src_tokenizer.get_vocab_size(), target_tokenizer.get_vocab_size()).to(device)

    # Creating the Tensorboard
    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    wandb_run_id = None
    model = DistributedDataParallel(model, device_ids=[config["local_rank"]])

    if config["preload"] != "":

        if config["preload"] == "latest":
            model_filename = get_latest_weights_file_path(config)
        else:
            model_filename = get_weights_file_path(config, int(config["preload"]))
        if model_filename is not None:
            print(f"GPU: {config['local_rank']} - Preloading model: {model_filename}")
            state = torch.load(model_filename)
            model.load_state_dict(state["model_state_dict"])
            sd = state["model_state_dict"]
            print("checkpoint keys:", list(sd.keys())[:10])

            # and compare to your model’s keys:
            print("model keys:", list(model.state_dict().keys())[:10])
            initial_epoch = state["epoch"] + 1
            optimizer.load_state_dict(state["optimizer_state_dict"])
            global_step = state["global_step"]
            wandb_run_id = state["wandb_run_id"]
            del state
        else:
            print(f"GPU: {config['local_rank']} - Could not fnd model to preload")

    if config["global_rank"] == 0:
        wandb.init(
            project="transformer-ddp",
            id=wandb_run_id,
            resume="allow",
            config=config
        )
    loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    if config["global_rank"] == 0:
        wandb.define_metric("global_step")
        wandb.define_metric("validation/*", step_metric="global_step")
        wandb.define_metric("train/*", step_metric="global_step")

    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()
        model.train()
        batch_trainer = tqdm(train_dataloader, desc=f"Processing epoch {epoch: 02d} on rank {config['global_rank']}", disable=config['local_rank']!=0)
        for batch in batch_trainer:
            encoder_input = batch["encoder_input"].to(device) # (Batch, Seq_len)
            # print(f"Encoder-input: {encoder_input}")
            decoder_input = batch["decoder_input"].to(device) # (B, Seq_len)
            # print(f"Decoder-input: {decoder_input}")
            encoder_mask = batch["encoder_mask"].to(device) # (B, 1, 1, Seq_len)
            decoder_mask = batch["decoder_mask"].to(device) # (B, 1, Seq_len, Seq_len)
            label = batch["label"].to(device) # (B, Seq_len)

            encoder_output = model.module.encode(encoder_input, encoder_mask) # (B, Seq_len, d_model)
            decoder_output = model.module.decode(encoder_output, decoder_input, encoder_mask, decoder_mask) # (B, Seq_len, d_model)
            projection_output = model.module.project(decoder_output) # (B, Seq_len, target_vocab_size)
            # projection_output = model(encoder_input, encoder_mask, decoder_input, decoder_mask)

            loss = loss_fn(projection_output.view(-1, target_tokenizer.get_vocab_size()), label.view(-1))
            batch_trainer.set_postfix({"loss": f"{loss.item():6.3f}", "global_step": global_step})

            if config["global_rank"] == 0:
                wandb.log({'train/loss': loss.item(), 'global step': global_step})
            # Log the loss
            writer.add_scalar("train_loss", loss.item(), global_step)
            writer.flush()

            # Backpropagation
            loss.backward()

            # Update the weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        if config["global_rank"] == 0:
            # Save the model after every epoch
            run_validation(model, src_tokenizer, target_tokenizer, writer, global_step, val_dataloader, lambda msg: batch_trainer.write(msg), device, config["seq_len"])

            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            print(f"[rank {config['global_rank']}] checkpoint path: {model_filename!r}")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                'wandb_run_id': wandb.run.id
            }, model_filename)
    
if __name__=="__main__":
    config = get_config()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=config["batch_size"])
    parser.add_argument('--num_epochs', type=int, default=config["num_epochs"])
    parser.add_argument('--lr', type=float, default=config["lr"])
    parser.add_argument('--seq_len', type=int, default=config["seq_len"])
    parser.add_argument('--d_model', type=int, default=config["d_model"])
    parser.add_argument('--lang_src', type=str, default=config["lang_src"])
    parser.add_argument('--lang_target', type=str, default=config["lang_target"])
    parser.add_argument('--model_folder', type=str, default=config["model_folder"])
    parser.add_argument('--model_basename', type=str, default=config["model_basename"])
    parser.add_argument('--preload', type=str, default=config["preload"])
    parser.add_argument('--tokenizer_path', type=str, default=config["tokenizer_path"])
    args = parser.parse_args()

    config.update(vars(args))

    config["local_rank"] = int(os.environ['LOCAL_RANK'])
    config["global_rank"] = int(os.environ['RANK'])

    assert config["local_rank"] != -1
    assert config["global_rank"] != -1

    if config["local_rank"] == 0:
        print("Configuration:")
        for key, value in config.items():
            print(f"{key:>20}: {value}")
    torch.cuda.set_device(config["local_rank"])
    init_process_group(backend="nccl")

    train_model(config)
    destroy_process_group()