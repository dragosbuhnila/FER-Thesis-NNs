import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.config import CONSOLE_OUTPUTS_PATH
from modules.misc import Tee

REDIRECT_OUTPUT = True
LOG_FILE = os.path.join(CONSOLE_OUTPUTS_PATH, f"{time.strftime('%Y%m%d-%H%M%S')}__hello_world__console_output.txt")

if REDIRECT_OUTPUT:
    log_dir = os.path.dirname(LOG_FILE)
    os.makedirs(log_dir, exist_ok=True)
    tee_instance = Tee(LOG_FILE)  # Create a single Tee instance
    sys.stdout = tee_instance
    sys.stderr = tee_instance  # Use the same instance for stderr

if __name__ == "__main__":
    print("=" * 200)  # Example long line
    mylist = list(range(100))
    print(f"My list from 0 to 10:   {mylist[0:10]}")
    print(f"My list from 0 to None: {mylist[0:None]}") 
    time.sleep(1)  # Simulate progress
    print("Another line of output.")
    time.sleep(1)
    print("Script finished.")
    raise ValueError("This is a test exception to check stderr redirection.")

    sys.stdout.close()  # Close the log file
    exit(0)