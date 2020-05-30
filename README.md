# BERTphone
This repo contains the implementation of the paper "BERTPHONE: Phonetically-aware Encoder Representations for Utterance-level Speaker and Language Recognition" in Pytorch
Paper: https://www.isca-speech.org/archive/Odyssey_2020/pdfs/93.pdf
Tutorial : https://www.youtube.com/watch?v=l-VCC0eaPMg


## Installation

I suggest you to install Anaconda3 in your system. First download Anancoda3 from https://docs.anaconda.com/anaconda/install/hashes/lin-3-64/
```bash
bash Anaconda2-2019.03-Linux-x86_64.sh
```
## Clone the repo
```bash
git clone https://github.com/KrishnaDN/BERTphone.git
```
Once you install anaconda3 successfully, install required packges using requirements.txt
```bash
pip iinstall -r requirements.txt
```

## Data Processing
This step convert raw data format suitable format for training.
Currently we support only TIMIT dataset and we are planning to extend it others as well in future
```
python datasets/timit.py --timit_dataset_root  /mnt/dataset/TIMIT --timit_save_root /mnt/dataset/processed_data --cmu_dict datasets/cmudict.dict --cmu_symbols datasets/cmudict.symbols
```

## Feature Extraction
This step extract features from audio files and store them in npy files in specified location
You can specify the feature type and feature dimension as the arguments in the following code.
```
python feature_extraction/feature_extraction.py --dataset_path  /mnt/dataset/processed_data --feature_store_path /mnt/dataset/Features --feature mfcc --feature_dim 13
```

## Create manifest files for training and testing
This step creates training and testing files for training
```
python create_meta_files.py --processed_data  /mnt/dataset/processed_data --meta_store_path meta/ 
```

## Training
This steps starts training the BERTphone model 
```
python training_BERTphone.py --training_filepath meta/training.txt --testing_filepath meta/testing.txt
                             --input_feat_dim 39 --num_phones 86 --num_heads 13 --num_layers 12 --lamda_val 0.1
                             --batch_size 32 --use_gpu True --num_epochs 100
```

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
For any queries contact : krishnadn94@gmail.com
## License
[MIT](https://choosealicense.com/licenses/mit/)