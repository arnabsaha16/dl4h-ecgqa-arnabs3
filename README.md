# Electrocardiogram–Language Model for Few-Shot Question Answering
This repository contains the code used for Electrocardiogram–Language Model for Few-Shot Question Answering with Meta Learning (published at CHIL 2025) and related implementation instructions. You can watch a brief project talk here: 

# Citation
@misc{tang2025electrocardiogramlanguagemodelfewshotquestion,
      title={Electrocardiogram-Language Model for Few-Shot Question Answering with Meta Learning}, 
      author={Jialu Tang and Tong Xia and Yuan Lu and Cecilia Mascolo and Aaqib Saeed},
      year={2025},
      eprint={2410.14464},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.14464}, 
}

# Motivation
Electrocardiogram (ECG) interpretation is a complex clinical task that requires specialized expertise to connect raw physiological signals with nuanced clinical queries expressed in natural language. However, the scarcity of labeled ECG data and the diversity of diagnostic questions make it difficult to build robust and adaptable systems. Replicating the ECG-QA paper is motivated by the need to validate a multimodal meta-learning approach that integrates pre-trained ECG encoders with large language models. By reproducing these experiments, we aim to assess the framework’s ability to generalize to unseen diagnostic tasks and confirm its potential to advance automated clinical reasoning in data-constrained scenarios. Also, both OpenAI models as well as others like google/llama-3-1.8B and meta/gemma-2-2b (latter were used in the paper) can be integrated with the meta learning framework to generate real-time responses for questions of different types asked for specific ECG waveforms with significant accuracy. 

# Data setup
Before starting on the modeling and experimentation aspects, I would like to highlight the prerequisites in terms of ECG waveforms and Question Answer related data access.
Data Extraction
1. PTB-XL: This ECG repository is available to access in multiple ways explained in the following Physionet website: https://physionet.org/content/ptb-xl/1.0.3/. The approach taken by me was to download the .ZIP file as a one-time effort, extract all data files from the same and store in local drive to run the preprocessing and modeling steps. For running some specific steps where the data volume was significant and I needed the processing to complete sooner, I ran them on a paid GPU Google Colab environment even though this is not mandatory. For this specific scenario, I connected to the data using the .ZIP file approach and extracted the files dynamically during runtime. 

2. MIMIC-IV-ECG: This is a larger ECG dataset also available in Physionet (https://www.physionet.org/content/mimic-iv-ecg/1.0/). The detailed steps to download and use this data is provided in the given link. The approach I used for this dataset is the same as for PTB-XL.

# Other prerequisites
This includes the following:
1. 

# Instructions to run ECG-QA experiments
1. Map file paths of ECGs with Question-Answer samples - For either PTB-XL or MIMIC-IV-ECG datasets, the first step is to map the ECG waveform files to their corresponding QA samples (or vice versa) since these two datasets are available in the public repositories separately. The command to run for this step looks as follows:
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import zipfile, os

# Extract PTB-XL dataset (Courtesy: https://physionet.org/content/ptb-xl/1.0.3/)
with zipfile.ZipFile("/content/DL4H/ECG_Datasets/ptb-xl-raw-dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("/content/datasets/")

!ls /content/datasets/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/

# Extract MIMIC-IV-ECG dataset (Courtesy: https://physionet.org/content/mimic-iv-ecg/1.0/)
with zipfile.ZipFile("/content/DL4H/ECG_Datasets/mimic-iv-ecg-raw-dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("/content/datasets/")

2. 
3. 

