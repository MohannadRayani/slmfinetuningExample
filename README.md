# SLM Fine Tuning Example:


# 1. To start you must first access Google Colab or other Services that provide online GPU access (ideally free):
## 📌 How to Create a Google Colab Account

Google Colab is a cloud-based Jupyter Notebook environment that allows you to run Python code for free. Follow these steps to create an account and start using Google Colab.

### **Step 1: Ensure You Have a Google Account**
Google Colab requires a Google account. If you don’t have one, create it by visiting [Google Sign-Up](https://accounts.google.com/signup).

### **Step 2: Open Google Colab**
1. Open your web browser and go to [Google Colab](https://colab.research.google.com/).
2. If you are not signed into your Google account, it will prompt you to log in.

### **Step 3: Accept Terms & Sign In**
1. If this is your first time using Google Colab, you might see a welcome message.
2. Accept any necessary permissions and terms of service.

### **Step 4: Create a New Notebook**
1. Click on **"File"** in the top menu.
2. Select **"New notebook"** to create a new Colab notebook.
3. A new Jupyter-style Python notebook will open.


# 2. Change Runtime to GPU in Google Colab

Google Colab allows you to utilize GPU acceleration for faster computations. Follow these steps to enable the **GPU runtime** in your Colab notebook.

## **Step 1: Open Google Colab**
1. Go to [Google Colab](https://colab.research.google.com/).
2. Open an existing notebook or create a new one by clicking **"File" > "New Notebook"**.

## **Step 2: Access Runtime Settings**
1. Click on **"Runtime"** in the top menu bar.
2. Select **"Change runtime type"** from the dropdown.

## **Step 3: Select GPU**
1. In the **"Runtime type"** pop-up window, locate the **"Hardware accelerator"** section.
2. Click on the dropdown menu and select **"GPU"**.
3. Click the **"Save"** button.

## **Step 4: Verify GPU Activation**
To confirm that the GPU is enabled, run the following Python command in a code cell:

```python
import torch

# Check if GPU is available
print("GPU Available:", torch.cuda.is_available())
```
# 3. Install required packages
## **Step 1: Run pip install on dependencies**
```python
!pip install transformers accelerate peft datasets bitsandbytes
```

# 4. Run code in copy_to_colab in the Google Colab environment

# 5. Run code in download.py in the Google Colab environment to download what you made in .zip







