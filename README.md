# LocAnalysis
My program "LocoA" uses a combination of .txt output files from the Trikinetics Locomotor Activity Monitor device and a meta dataset that includes information on each test subject to create plots, perform statistical analyses, and generate predictions.


### In development

LocAnalysis is still under development. If you would like to install the program locally to help work on the code, please follow the instructions below:

```
# stay tuned, more to be released soon...
# conda install [list dependencies here...] -c conda-forge ...

conda install pandas numpy matplotlib scikit-learn umap-learn mpld3 scipy -c conda-forge
conda install -c jmcmurray os

git clone https://github.com/CLan2021/LocoA.git

cd ./LocoA

pip install -e .
```
