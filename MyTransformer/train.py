import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_config, get_weights_file_path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from tqdm import tqdm
import warnings

def get_all_sentences(ds,lang):
    # Here we will keep all the sentences from our dataset in the language "lang"
    for item in ds :
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    # First, we define the path where the tokenizer should be saved
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    # If the tokenizer doesn't exist, we must create it
    if not Path.exists(tokenizer_path):
        # Here we define a Tokenizer that will split a text in words and that will use the token UNK when he doesn't know a word
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        # Then we explicitely tell the tokenizer that the words will be separated by spaces
        tokenizer.pre_tokenizer = Whitespace()
        # Until this point, the tokenizer we created doesn't know the words of the vocabulary 
        # As a matter of fact, he can't give to each word a token, we have to train it
        # The WordLevelTrainer will go over the dataset and will create for each word a token 
        # However, we used min_frequency = 2 meaning that only words appearing at least 2 times deserve a token
        # And we will create specific tokens for padding, start of sequence and end of sequence
        trainer = WordLevelTrainer(special_tokens=['[UNK]','[PAD]','[SOS]','[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
        # Finally, we save the tokenizer, so that we can reuse it in the future
        tokenizer.save(str(tokenizer_path))
    else : 
        # The tokenizer already exist so we load it
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config) :
    # Here we will load the dataset from hugging face
    ds_raw = load_dataset("opus_books",f"{config['lang_src']}-{config['lang_tgt']}",split='train')

    # Then we will build the tokenizers
    tokenizer_src = get_or_build_tokenizer(config,ds_raw,config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config,ds_raw,config['lang_tgt'])

    # Dataset split into train and test
    train_ds_size = int(0.9 * len(ds_raw))
    test_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, test_ds_raw = random_split(ds_raw,[train_ds_size,test_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    test_ds = BilingualDataset(test_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # We will check the max length of the sentences in the src language and the tgt language
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw :
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src,len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence : {max_len_src}")
    print(f"Max length of target sentence : {max_len_tgt}")

    # Then we define the DataLoaders for the training and the test
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'],shuffle=True)
    test_dataloader = DataLoader(test_ds,batch_size=1,shuffle=True)

    return train_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len) : 
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True,exist_ok=True)

    train_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])    

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload'] :
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_state']
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator : 
            encoder_input = batch['encoder_input'].to(device) # shape (batch_size, seq_len)
            decoder_input = batch['decoder_input'].to(device) # shape (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # shape (batch_size, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # shape (batch_size, 1, seq_len, seq_len )

            encoder_output = model.encode(encoder_input,encoder_mask)
            decoder_output = model.decode(decoder_input, encoder_output,encoder_mask, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)

            loss = loss_fn(proj_output.view(-1,tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss" : f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
        
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'global_step' : global_step
        }, model_filename)

if __name__ == '__main__' :
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)