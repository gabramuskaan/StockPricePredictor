import streamlit as st
import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.abspath("app"))

# Import the main function from app.py
from app import main

# Run the app
if __name__ == "__main__":
    main()git add .
git commit -m "Prepare for Streamlit deployment"
git push
