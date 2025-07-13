import logging


def setup_logging():
    # Configure basic settings
    logging.basicConfig(
        level=logging.INFO,  # Minimum log level to display
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
        handlers=[
            logging.FileHandler("app.log"),  # Log to a file
            logging.StreamHandler()  # Also log to console
        ]
    )

def get_logger(name=None):
    return logging.getLogger(name)