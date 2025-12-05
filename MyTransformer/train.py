import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_config, get_weights_file_path

from datasets import load_dataset
import sentencepiece as spm

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
        with open(f"corpus_{lang}.txt", 'w', encoding='utf-8') as f:
            for sentence in get_all_sentences(ds, lang):
                f.write(sentence + '\n')

        # We use Sentence Piece, this tokenizer is way better than WordLevel Tokenizer
        # Because WordLevel Tokenizer would have associated to each word a token
        # Here the sentence piece tokenizer will give tokens to parts of words.
        # For example, if we consider the words eat, eating, sleep, sleeping.
        # A word level tokenizer would give us 4 tokens. However, the tokenizer that we will use would give us three tokens : eat, sleep, ing. 
        # This will allow better handling of rare words and improves generalization.
        spm.SentencePieceTrainer.Train(
            input=f'corpus_{lang}.txt',
            model_prefix=f'bpe_{lang}',
            vocab_size=16000,
            model_type='bpe',
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3
        )
    return spm.SentencePieceProcessor(model_file=config['tokenizer_file'].format(lang))

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
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']], out_type=int)
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']], out_type=int)
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

def greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device) :
    # Set the begin and end tokens
    sos_idx = tokenizer_tgt.bos_id()
    eos_idx = tokenizer_tgt.eos_id()

    # Encode the sentence to translate
    encoder_output = model.encode(encoder_input, encoder_mask)

    # At the beginning, we didn't translate anything, so we just have the beginning token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(encoder_input).to(device)
    
    # We use while True because we will break during the loop
    while True : 
        # If we already produced max_len tokens, then we stop
        if decoder_input.size(1) == max_len :
            break

        # We create the mask depending on the size of the sentence being generated
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)

        # We obtain for each of the words of the sentence being generated the embeddings
        decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)

        # We are only interested by the next token, so we will pick the last embedding (last word)
        proj_output = model.project(decoder_output[:,-1])
        # We get the token corresponding to the next word
        _, next_word = torch.max(proj_output, dim=1)
        # We add the new token at the end of the sentence
        decoder_input = torch.cat([decoder_input,torch.empty(1,1).type_as(encoder_input).fill_(next_word.item()).to(device)],dim=1)

        # If it's the end token, we stop
        if next_word.item() == eos_idx :
            break

    return decoder_input.squeeze(0)

def run_test(model, test_dataloader, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples = 5):
    # Set the model to be evaluated
    model.eval()

    count = 0

    source_texts = []
    expected = []
    predicted = []

    console_width = 80
    
    # Tell torch that there are no gradients to calculate now
    with torch.no_grad() : 
        # Goes through each sentence
        for batch in test_dataloader :
            count += 1
            # Sentence to translate
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for test"

            # more details in the greedy decode function
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().tolist())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            print_msg('-'*console_width)
            print_msg(f"SOURCE: {source_text}")
            print_msg(f"TARGET: {target_text}")
            print_msg(F"PREDICTED: {model_out_text}")

            if count == num_examples:
                break
            
def train_model(config):
    # If you have a GPU, it will use it, else CPU will have to do it
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True,exist_ok=True)

    # Sets dataloaders and tokenizers
    train_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    # Initialize the transformer models, and puts it in the defined device (your GPU I hope)
    model = get_model(config, tokenizer_src.get_piece_size(), tokenizer_tgt.get_piece_size()).to(device)

    # Output text
    writer = SummaryWriter(config['experiment_name'])    

    # We use the Adam Method to update the weights of the model
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # Initialize or Preload a state
    initial_epoch = 0
    global_step = 0
    if config['preload'] :
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    
    # We define the loss, we will use cross entropy loss because it is the most used for classification / language modeling
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.pad_id(), label_smoothing=0.1).to(device)

    # We will run through various epochs
    for epoch in range(initial_epoch, config['num_epochs']):
        # We set the model for training
        model.train()
        # Output bar
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        # We will go over batches (groups) of sentences
        for batch in batch_iterator : 
            # Sentence to translate
            encoder_input = batch['encoder_input'].to(device) # shape (batch_size, seq_len)
            # Translated Sentence
            decoder_input = batch['decoder_input'].to(device) # shape (batch_size, seq_len)
            # Mask padding
            encoder_mask = batch['encoder_mask'].to(device) # shape (batch_size, 1, 1, seq_len)
            # Mask padding and future words 
            decoder_mask = batch['decoder_mask'].to(device) # shape (batch_size, 1, seq_len, seq_len )

            # Goes through every step of the encoder part
            encoder_output = model.encode(encoder_input,encoder_mask)
            # Goes through every step of the decoder part
            decoder_output = model.decode(decoder_input, encoder_output,encoder_mask, decoder_mask)
            # Predicts for each word the next one
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)

            # Calculate the loss between the predicted words and the label
            loss = loss_fn(proj_output.view(-1,tokenizer_tgt.get_piece_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss" : f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Calculate the gradients 
            loss.backward()

            # Modify weights with the Adam algorithm
            optimizer.step()
            # Sets the gradients to zero
            optimizer.zero_grad()

            global_step += 1

        # Tries the model at the end of each epoch
        run_test(model, test_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        
        # Saving the model parameters
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