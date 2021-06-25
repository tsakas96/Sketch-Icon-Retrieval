# Sketch-Icon-Retrieval

# 1)Setup a virtual environment using conda - We use vscode
## Command for creating and updating the requirements.txt
pip freeze > requirements.txt

## Create Conda virtual Environment
conda create --prefix C:\programming\Dataset python=3.8.5

## Activate Conda Environment
Open code from the anaconda terminal
Ctrl+Shift+P and search “Terminal”.
Choose “Terminal: Create New Integrated Terminal (In Active Workspace)

Or from anaconda cmd (better option)
conda activate C:\programming\ThesisModels

## Deactivate Conda Environment
conda deactivate

## How to run python
Both interpreter (bottom left) and Jupyter Kernel (top right) should be in path "C:\programming\ThesisModels\python.exe"

# Important Notes
For the Jupyter notebook to work in vscode, anaconda should be install because it has all the required
packages to run.
The python files are better to be in a different folder than the environment. In that way, github can be used easier to store only the python files and not the entire virtual environment.

tensorflow version 2.3.0

# 2) Run Notebooks on Google Colab or locally
To run the notebooks on google colab or locally, the imported util and model files should be in the same directory as the notebook.
The models are in the models folder while the utils files are in the utils folder and both folders are in Scripts.

In the Optimized folder we include three notebooks which are also included in the triplet folder. We use the tf.data.Dataset to read faster the data from memory.

In the Results-Cluster we provide all the models we trained
