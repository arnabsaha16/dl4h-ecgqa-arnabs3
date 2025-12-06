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
1. PTB-XL: This ECG repository is available to access in multiple ways explained in the following Physionet website: https://physionet.org/content/ptb-xl/1.0.3/. The approach taken by me was to download the .ZIP file as a one-time effort, extract all data files from the same and store in local drive to run the preprocessing and modeling steps. For running some specific steps where the data volume was significant and I needed the processing to complete sooner, I ran them on a paid GPU Google Colab environment even though this is not mandatory. For this specific scenario, I connected to the data using the .ZIP file approach and extracted the files dynamically during runtime. Required python packages to be imported to the program for following this approach are: requests, zipfile and io.

2. MIMIC-IV-ECG: This is a much larger ECG dataset also available in Physionet (https://www.physionet.org/content/mimic-iv-ecg/1.0/). The detailed steps to download and use this data is provided in the given link. The approach I used for this dataset is the same as for PTB-XL.

Note: MIMIC-IV-ECG dataset is quite large and failed to run after mapping step on all occasions, so I created a custom logic to use only a small subset of the data files for processing. The code has been uploaded to this repository for reference. 

# Other system prerequisites
This includes the following:
* PyTorch version >= 1.5.0
* Python version >= 3.6, and <= 3.9
* For training new models, you'll also need an NVIDIA GPU. For reference, some of the processing done to replicate the results of this paper was performed using NVIDIA® A100 and NVIDIA® T4 single GPUs on Google Colab environment.
* Install and import the following python packages:
      $ pip install tqdm pandas
      $ pip install --upgrade matplotlib
      $ pip install transformers==4.30.2
      $ pip install wandb
      $ pip install wfdb
      $ import pandas as pd
* Some environment variables also needed to be set to specific values:
      $ os.environ["TOKENIZERS_PARALLELISM"] = "false"
* Clone "ecg-qa" repository to local environment
      $ git clone https://github.com/Jwoo5/ecg-qa
* Clone and install "fairseq-signals" from source and develop locally:
      $ git clone https://github.com/Jwoo5/fairseq-signals
      $ cd fairseq-signals
      $ pip install --editable ./
* Importing pretrained checkpoints (around 1 GB) related to upperbound experiments for W2V+CMSC+RLM and for LLM modeling using OpenAI models like gpt-4.1. These are readily available in the Hugging Face repository (https://huggingface.co/wanglab/ecg-fm) related to one of the reference papers () provided within the above Github repositories.  

*Note:* Referring to the last point above, there is dependency on 2 Github repositories: Jwoo5/ecg-qa and Jwoo5/fairseq-signals for this paper. The same will need to be cloned to the runtime environment used for running the implementation steps explained here. However, only the fairseq-signals codebase needs installation that too in 'editable' mode installation, which means the package is linked to your local source tree. Any changes you make to the code after installation are immediately reflected without needing to reinstall.

*Note*: Please refer the README.md file of the official Github repository for this research paper - https://github.com/Jwoo5/ecg-qa (along with the README file added in the fairseq-signals repository discussed above) for the following details:
- Dataset structure of the 2 repositories, both for the ECG waveforms and Question-Answer samples
- Codelines to read a sample ECG data as well as already mapped ECG IDs to QA samples pre-loaded into the ECG-QA repository. The ECG files are mapped to both the template and paraphrased versions of the Questions in the QA dataset and the resulting mapping JSON files are stored in separate folders.
- Most of the preprocessing, experimentation and LLM modeling steps listed below, although some of the codes needed to be modified (copies of those programs uploaded in this repository) to make it compatible with the Python version and its dependencies available as of the documentation of this paper.

# Instructions to run ECG-QA experiments
1. Map file paths of ECGs with Question-Answer samples - For either PTB-XL or MIMIC-IV-ECG datasets, the first step is to map the ECG waveform files to their corresponding QA samples (or vice versa) since these two datasets are available in the public repositories separately. The command to run for this step looks as follows:
*PTB-XL:*
```shell script
      $ python mapping_ptbxl_samples.py ecgqa/ptbxl \
          --ptbxl-data-dir $ptbxl_dir \
          --dest $dest_dir
```
*MIMIC-IV-ECG:*
```shell script
      $ python mapping_mimic_iv_ecg_samples.py ecgqa/mimic-iv-ecg \
          --mimic-iv-ecg-data-dir $mimic_iv_ecg_dir \
          --dest $dest_dir
```
As same with the above, you can also process the template version of ECG-QA by passing `ecgqa/mimic-iv-ecg/template`.  
`$mimic_iv_ecg_dir` should be set to the root directory of the MIMIC-IV-ECG dataset which contains `files/` directory and `record_list.csv` file. If you do not specify this argument, the script will automatically download the required MIMIC-IV-ECG data to the cache directory ($HOME/.cache/ecgqa/mimic-iv-ecg).  
Note that `$dest_dir` is set to `./output/mimic-iv-ecg` by default.




