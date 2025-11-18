import yaml
import os 

def load_config(config_path='config.yaml'):
    """Carrega o arquivo de configuração YAML."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config["model"]["name"] = os.environ.get("MODEL_NAME", config["model"]["name"])
    return config