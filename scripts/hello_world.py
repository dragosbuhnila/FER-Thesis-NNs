import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time

from modules.config import CONSOLE_OUTPUTS_PATH



REDIRECT_OUTPUT = True
LOG_FILE = os.path.join(CONSOLE_OUTPUTS_PATH, f"{time.strftime('%Y%m%d-%H%M%S')}__hello_world__console_output.txt")

if REDIRECT_OUTPUT:
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout = open(LOG_FILE, "w")
    sys.stderr = sys.stdout

if __name__ == "__main__":
    print("hello_world")