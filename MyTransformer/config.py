from pathlib import Path

def get_config():
    return {
        "batch_size" : 12,
        "num_epochs" : 40,
        "lr" : 10**-4,
        "seq_len" : 500,
        "d_model" : 128,
        "lang_src": "es",
        "lang_tgt": "fr",
        "model_folder": "weights",
        "model_filename": "tmodel_",
        "preload" : None,
        "tokenizer_file": "bpe_{0}.model",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_filename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)