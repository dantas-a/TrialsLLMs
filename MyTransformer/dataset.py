from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset) :
    def __init__(self, ds,tokenizer_src,tokenizer_tgt, src_lang,tgt_lang,seq_len) -> None:
        super().__init__()

        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')],dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')],dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')],dtype=torch.int64)

    def __len__(self) :
        return len(self.ds)
    
    def __getitem__(self,index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Here we encode the src text with the src tokenizer and the tgt text with the tgt tokenizer
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # It is important to calculate the difference between the max length of a sequence and length of the actual sequence
        # It allows us to check if the sequence isn't too long and will allow the padding
        # The goal of the padding is to make sure that all the sequences have the same length
        # It is important to remove 2 from the result so that we can add the SOS and EOS tokens
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        # Here we only remove 1, because all the generated sequences only need the SOS token
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # If the sequence is too long, we must raise an error
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0 :
            raise ValueError('Sentence is too long')
        
        # Here add the SOS token at the beginning of the sequence, the EOS at the end of the sequence and the pad after to fill
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens,dtype=torch.int64)
            ]
        )

        # Here we add the SOS token at the beginning and after the sequence we pad
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)                 
            ]
        )
        
        # We use the EOS token after the sequence and then we pad
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
    
        # Check if every tensor has the expected length
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # We will return the encoder_input, the decoder_input, the encoder_mask that has to hide all the padding tokens,
        # the decoder mask that hides all the padding tokens and the words after the word we are looking at,
        # we return the label and then the src_text and tgt_text for visualization
        return {
            "encoder_input" : encoder_input,
            "decoder_input" : decoder_input,
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label" : label,
            "src_text" : src_text,
            "tgt_text" : tgt_text
        }
        
def causal_mask(size) :
    # We create a upper triangular matrix with ones (diagonal=1 makes sure the diagonal has 0's)
    mask = torch.triu(torch.ones(1,size,size),diagonal=1).type(torch.int)
    # Then we mark as True where there are 0's and False everywhere else
    # Which gives us a lower triangular matrix (diagonal included)
    # Here the queries are on the lines and the key on the columns, as a matter of fact, a query can't see the answer from a key which is after in the sequence
    return mask == 0