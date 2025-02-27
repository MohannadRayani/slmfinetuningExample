import shutil

# Define the path where the model is saved
model_dir = "tiny-llama-lora-adapter"

# Zip the model directory
shutil.make_archive(model_dir, 'zip', model_dir)

print("Model zipped successfully.")

from google.colab import files
files.download("tiny-llama-lora-adapter.zip")