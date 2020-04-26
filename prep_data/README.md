# Data preparation 

## Download the data 
Use the download scripts provided in the official [github repo](https://github.com/cvdfoundation/google-landmark) to download the data and csv files.

Image files are extracted and stored in nested folders. 

## Organize the data 

Run the below script to organize the data into PyTorch [ImageFolder format](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder).

e.g. root_folder/<landmark_id>/<img_id>.jpg

```bash
$ python organize_dataset.py -h
usage: Organize the data into Image Folder format [-h] [-c CSV_FILE]
                                                  [-i ROOT_DIR]
                                                  [-o OUTPUT_DIR]
                                                  [-e EXTENSION]

optional arguments:
  -h, --help            show this help message and exit
  -c CSV_FILE, --csv_file CSV_FILE
                        csv file containing instance to label mapping
  -i ROOT_DIR, --root_dir ROOT_DIR
                        root image folder
  -o OUTPUT_DIR, --output_idr OUTPUT_DIR
                        output folder
  -e EXTENSION, --extension EXTENSION
                        file extension. default: jpg
```

## Clean the data 

Google Landmark Dataset v2 is a large dataset with 4,132,914 images in the train set and 203,095 landmarks. Use the below script to clean the data following the similar procedures described in this [report](https://arxiv.org/pdf/1906.11874.pdf).
The cleaned version is stored in a csv file (img_id, landmark_id).


```bash
$ python clean_data.py -h
usage: Clean Google Landmark Dataset v2 using ResNet 50 features
       [-h] [-c CSV_FILE] [-o OUTPUT_FILE] [-l LABEL_COLUMN] [-f FEATURE_FILE]
       [-d DATA_DIR] [-e EXTENSION]

optional arguments:
  -h, --help            show this help message and exit
  -c CSV_FILE, --csv_file CSV_FILE
                        csv file containing instance to label mapping
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        output csv file name for the clean training data
  -l LABEL_COLUMN, --label_column LABEL_COLUMN
                        column name storing label id
  -f FEATURE_FILE, --feature_file FEATURE_FILE
                        output npy file name for the extracted ResNet features
                        of the image
  -d DATA_DIR, --data_dir DATA_DIR
                        root image folder
  -e EXTENSION, --extension EXTENSION
                        file extension. default: jpg
```