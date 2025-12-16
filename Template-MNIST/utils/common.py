import yaml, logging

def load_config(yaml_file_path: str) -> dict:
    with open(yaml_file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)
    
class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",   # cyan
        logging.INFO: "\033[32m",    # green
        logging.WARNING: "\033[33m", # yellow
        logging.ERROR: "\033[31m",   # red
        logging.CRITICAL: "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record):
        color = ColorFormatter.COLORS.get(record.levelno, "")
        msg = super().format(record)
        return f"{color}{msg}{ColorFormatter.RESET}"

logging_handler = logging.StreamHandler()
logging_handler.setFormatter(ColorFormatter("[%(levelname)s] %(message)s"))