import logging
import sys

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("logs.log", mode='w'),
                        logging.StreamHandler(sys.stdout)
                    ])