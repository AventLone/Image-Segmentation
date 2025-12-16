import yaml

def load_config(yaml_file_path: str) -> dict:
    with open(yaml_file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)