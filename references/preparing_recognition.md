First make sure development version of pytorch version of doctr is installed from the h2o_ocr branch with: 

pip install -e doctr/.[torch]
pip install fastprogress
pip install wandb

Second install aria2c with:

sudo apt install aria2

pip install thefuzz

cd doctr

This will help you download all of the datasets via torrent much faster

Now download synthtext with:

aria2c SynthText-2dba9518166cbd141534cbf381aa3e99a087e83c.torrent

Download mjsynth with 
aria2c mjsynth.tar.gz-3d0b4f09080703d2a9c6be50715b46389fdb3af1.torrent 

Download imgur handwriting dataset by using the repo from here https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset:
git clone https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset.git 
python3 IMGUR5K-Handwriting-Dataset/download_imgur5k.py --dataset_info_dir IMGUR5K-Handwriting-Dataset/dataset_info/ --output_dir IMGUR5K-Handwriting-Dataset/dataset/

mjsynth needs to be unzipped
tar -xvf mjsynth.tar.gz

cd SynthText
unzip SynthText.zip
cd ..

rename mjsynth data
mv ./mnt/ramdisk ./mnt/images

Now that the 3 datasets are downloaded they will need to be prepped with the data prep scripts from doctr/datasets

python3 prep_mjsynth.py
python3 prep_imgur.py
python3 prep_synthtext.py

All of the datasets come in different formats and this will standardize them to a single format compatible with the doctr recognition datasets where it is simply a directory with images and a json file with the path to the images as the keys and their labels as the values
