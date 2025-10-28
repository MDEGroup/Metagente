import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_HOST = os.environ.get("MONGODB_HOST")
MONGODB_DATABASE = os.environ.get("MONGODB_DATABASE")
MONGODB_COLLECTION = os.environ.get("MONGODB_COLLECTION")
