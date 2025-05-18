import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, src_tokenizer, target_tokenizer, src_language, target_language, seq_len):
        super().__init__()
        self.src_tokenizer = src_tokenizer
        self.target_tokenizer = target_tokenizer
        self.src_language = src_language
        self.target_language = target_language
        self.seq_len = seq_len
        self.ds = ds

        self.pad_token = torch.tensor([target_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)
        self.eos_token = torch.tensor([target_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.sos_token = torch.tensor([target_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]

        # Getting the src and target inputs
        src_input = src_target_pair["translation"][self.src_language]
        target_input = src_target_pair["translation"][self.target_language]

        # Converting to encoder input tokens and decoder input tokens

        enc_input_tokens = self.src_tokenizer.encode(src_input).ids
        dec_input_tokens = self.target_tokenizer.encode(target_input).ids

        # calculating the number of padding tokens for each

        enc_num_padding = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding < 0 or dec_num_padding < 0:
            raise ValueError("Sentence is too long")
        
        # building the enconder inputs and decoder inputs with EOS, SOS and padding

        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding, dtype=torch.int64)
        ], dim=0)

        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding, dtype=torch.int64)
        ], dim=0)

        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding)
        ], dim=0)

        assert encoder_input.shape[0] == self.seq_len
        assert decoder_input.shape[0] == self.seq_len
        assert label.shape[0] == self.seq_len

        return {
            "encoder_input": encoder_input, # (Seq_len)
            "decoder_input": decoder_input, # (Seq_len)
            "label": label, # (Seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1, 1, Seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.shape[0]),
            "src_text": src_input,
            "target_text": target_input
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0





