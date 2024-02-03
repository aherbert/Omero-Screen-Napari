from dotenv import load_dotenv
import os

# Assuming this block is intended to choose between .env and .localenv
try:
    # Specify the path if .localenv is not in the current directory
    dotenv_path = '.localenv'  # or the full path to your .localenv file
    load_dotenv(dotenv_path=dotenv_path)
except IOError:
    # Fallback to default .env or handle the error
    load_dotenv()  # This will default to loading .env if present

# After loading, retrieve your environment variables
username = os.getenv("USERNAME")
password = os.getenv("PASSWORD")
host = os.getenv("HOST")

print(username, password, host)