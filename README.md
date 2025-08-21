DASH is a molecular dynamics software to sample functional-state conformations of proteins from ground-state conformations
##
Environmental Setup
```
conda create -n dash python==3.9 openmm==8.0.0 openmm-torch==1.0 pytorch==1.13.1 -c conda-forge
pip install prody matplotlib scikit-learn numba tqdm torch-geometric torch_scatter
pip install wheel
pip install pyemma
conda activate dash
```
If the you failed to use the above configurations, try some alternations, including :

python==3.8 pytorch==1.11.0 openmm==8.0.0 openmm-torch==1.0

python==3.11.0 pytorch==2.3.0 openmm==8.1.1 openmm-torch==1.4

Or you can try removing some version restrictions of above packages so that conda could find the appropriate versions that fit your hardware.
##
Install DASH
```
git clone https://github.com/JinyinZha/DASH.git
cd DASH
python install_DASH.py
```
##
Useage

Input files for simulations in this article are all in the folder ```input_files/```. Let's take an example.
```
cd input_files/inactive2active
../../bin/DASH -i smallG_inactive.in > smallG.out &
#This script would take 2-3 days
```
If you want to perform dimension reductions of the cases in the article, you should download the data first.
```
cd data
wget https://zenodo.org/record/16917262/files/train_data.zip
unzip train_data.zip
```
Then you can run the dimension reductions.
```
cd train_data/trp_cage
../../../bin/DASH_Train SplitContrastMAE train.in CA
```
More information of the useages could be found in ```tutorial/```
