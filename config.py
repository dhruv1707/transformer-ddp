from pathlib import Path

def get_config():
    return {
        "lang_src": "en",
        "lang_target": "it",
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_path": "tokenizer_{0}.json",
        "experiment_name": "runs/model"
    }

def get_weights_file_path(config, epoch):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)
