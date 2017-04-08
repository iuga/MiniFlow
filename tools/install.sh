# 1. Create the virtual environment
conda remove --name miniflow --all
conda create --yes --name miniflow numpy scikit-learn click pytest jupyter
# 2. Activate the environment
source activate miniflow
# 3. Install PIP dependencies
pip install pytest-pythonpath
