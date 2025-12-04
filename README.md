# cse168-project
Team FantasticFoursome
## How to Run

1. Download the Eurosat_MS dataset from the following link: https://zenodo.org/records/7711810#.ZAm3k-zMKEA 
2. Move it to under the 'cse168-project' repo folder.
3. cd into 'cse168-project' folder.
4. then run `unzip EuroSAT_MS.zip` . This should fully extract all the images into a new folder with respective directories.
5. activate the conda env using the accompanying `eurosat.yaml` file.
6. Now train `python train.py`
7.   Please note: number of workers is currently set to 32, if running locally then it be best to decrease this number down, maybe 4(?).

