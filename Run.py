import subprocess
import sys
import os

# app.py ka path
app_path = os.path.join(os.path.dirname(__file__), "Final_Code.py")

# Streamlit ko subprocess me run karo
# shell=True ensures proper execution on Windows
subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], shell=True)