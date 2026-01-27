import logging
import os
from datetime import datetime

log_file=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logs_dir=os.path.join(os.getcwd(),"logs")
os.makedirs(logs_dir,exist_ok=True)
log_file_path=os.path.join(logs_dir,log_file)

logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

