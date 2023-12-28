rm -rf /env/

module load conda

conda create -p "$PWD"/env/ -y
conda activate "$PWD"/env/

conda install python=3.8 -y

which python
python --version
pip install tensorflow==2.9.0 tensorflow-probability tensorflow-determinism numpy matplotlib scipy pandas psutil scikit-learn datetime

conda env export > requirements.yml