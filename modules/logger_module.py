# logger_module.py
import logging
import sys
import numpy as np
import os
import torch

def get_logger(name: str = "train", log_file: str = "./logfile.log") -> logging.Logger:
    """
    Creates and returns a logger with the given name.

    Args:
    - name (str): Name of the logger. Defaults to "train".
    - log_file (str): Path to the log file where logs should be written.

    Returns:
    - logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # StreamHandler for console logging
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        logger.addHandler(console_handler)

        # FileHandler for file logging
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        logger.addHandler(file_handler)

    return logger


def log_args(args, logger: logging.Logger):
    """
    Logs the arguments.

    Args:
    - args: Parsed arguments.
    - logger (logging.Logger): Configured logger.
    """
    logger.info(" ====== Arguements ======")
    for k, v in vars(args).items():
        logger.info(f"{k:25}: {v}")

def setup_seed(seed: int, logger: logging.Logger):
    """
    Sets up the random seed for reproducibility.

    Args:
    - seed (int): Random seed value.
    - logger (logging.Logger): Configured logger.
    """
    logger.info(f"[+] Set Random Seed to {seed}")
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
