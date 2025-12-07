# Electrocardiogram–Language Model for Few-Shot Question Answering - Replication of research paper
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
1. PTB-XL: This ECG repository is available to access in multiple ways explained in Physionet (https://physionet.org/content/ptb-xl/1.0.3/). The approach taken by me was to download the .ZIP file as a one-time effort, extract all data files from the same and store in local drive to run the preprocessing and modeling steps. For running some specific steps where the data volume was significant and I needed the processing to complete sooner, I ran them on a paid GPU Google Colab environment even though this is not mandatory. For this specific scenario, I connected to the data using the .ZIP file (uploaded in Google Drive) approach and extracted the files dynamically during runtime.

      $ from google.colab import drive drive.mount('/content/drive', force_remount=True)
      $ with zipfile.ZipFile("/content/drive/MyDrive/DL4H/ECG_Datasets/ptb-xl-raw-dataset.zip", 'r') as zip_ref:
               zip_ref.extractall("/content/datasets/")

3. MIMIC-IV-ECG: This is a much larger ECG dataset also available in Physionet (https://www.physionet.org/content/mimic-iv-ecg/1.0/). The detailed steps to download and use this data is provided in the given link. The approach I used for this dataset is the same as for PTB-XL.

      $ from google.colab import drive drive.mount('/content/drive', force_remount=True)
      $ with zipfile.ZipFile("/content/drive/MyDrive/DL4H/ECG_Datasets/mimic-iv-raw-dataset.zip", 'r') as zip_ref:
               zip_ref.extractall("/content/datasets/")

Note: MIMIC-IV-ECG dataset is quite large and failed to run after mapping step on all occasions, so I created a custom logic to use only a small subset of the data files for processing. The code has been uploaded to this repository for reference. 

# Other system prerequisites
This includes the following:
* PyTorch version >= 1.5.0
* Python version >= 3.6, and <= 3.9
* For training new models, you'll also need an NVIDIA GPU. For reference, some of the processing done to replicate the results of this paper was performed using NVIDIA® A100 and NVIDIA® T4 single GPUs on Google Colab environment.
* Install and import the following python packages:
- $ pip install tqdm pandas
- $ pip install --upgrade matplotlib
- $ pip install transformers==4.30.2
- $ pip install wandb
- $ pip install wfdb, wandb, openai
- $ import pandas as pd
- $ import zip, os (optional, required if connecting to online storage e.g. Google Drive for loading raw data in local drive)
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
*Notes:*
* $ptbxl_dir and $mimic_iv_ecg_dir refer to the root folder of the respective dataset (after download) in the current system used for processing this step.
* $dest_dir in both cases refers to the location in which the JSON files which contain the mapping between ECG IDs and the Question Answer templates is to be stored.
* Sampling is done to split the available data into training, validation and test datasets. The subsequent steps will consider this grouping for further processing.

2. Pre-process ECG-QA dataset.
```shell script
      $ python fairseq_signals/data/ecg_text/preprocess/preprocess_ecgqa.py /path/to/ecgqa \
            --dest /path/to/output \
            --apply_paraphrase
```

*Notes:*
* /path/to/ecgqa is the same location as $dest_dir discussed in previous step.
* If you run with --apply_paraphrase, the scripts will process the paraphrased version of ECG-QA dataset. Otherwise, it will process the template version.
* /path/to/output is the location in which the .mat (Microsoft Access Table) files corresponding to the ECG data are generated, along with one .tsv (Tab separated values) file for each of the 3 groups where all the files are listed along with information on the count of samples across a specific count of leads is captured. The directory path for all the files is captured in the first row for the next steps.  

3. Run experiments.
```shell script
      $ fairseq-hydra-train task.data=/path/to/output/paraphrased \
            model.num_labels=$num_labels \
            --config-dir /fairseq-signals/examples/scratch/ecg_question_answering/$model_name \
            --config-name $model_config_name
```

*Notes:*
* $num_labels: the number of answers specified in answers.csv (103 for ptb-xl version and 187 for mimic-iv-ecg version). The answer *none* is not considered.
* $model_name: the name of the ECG-QA model (e.g., ecg_transformer)
* $model_config_name the name of the configuration file (e.g., base)


# Instructions to run upperbound experiments
1. Same mapping step mentioned as step 1 in the above 'Run QA Experiments' section.

2. Pre-process ECG-QA dataset to be compatible with upperbound experiments.

      $ python fairseq_signals/data/ecg_text/preprocess/preprocess_ecgqa_for_classification.py /path/to/ecgqa \
            --dest /path/to/output

*Notes:*
* /path/to/ecgqa is the root location (folder) of the mapping step output.
* /path/to/output is the output path where all the composite metadata about the ECG sample after preprocessing is stored. The naming convention is different from the preprocessed files in the QA experiments, in the sense that these files generally capture the mapping between ECG .mat files and the various conditions (groups) into which the given file can be classified into. For example, name of 10000_0_1_2_4_5_10_11.mat indicate the following:
- 10000 → Usually the record ID or patient/sample identifier. In preprocessing pipelines, each ECG recording is assigned a unique integer ID.
- 0_1_2_4_5_10_11 → These are the class labels (diagnostic codes) associated with that ECG.
- During preprocessing for classification, multiple labels are often concatenated with underscores to indicate that this sample belongs to several classes simultaneously.
- For example, 0, 1, 2, 4, 5, 10, 11 might correspond to specific diagnostic categories (e.g., myocardial infarction, conduction disturbance, hypertrophy, etc.) depending on the label mapping file

3. For W2V+CMSC+RLM:
```shell script
      $ fairseq-hydra-train task.data=/path/to/output \
            model.num_labels=$num_labels \
            model.model_path=/path/to/checkpoint.pt \
            --config-dir /fairseq-signals/examples/w2v_cmsc/config/finetuning/ecg_transformer/grounding_classification \
            --config-name base_total
```
*Notes:*
* The above is a training command for running the W2V+CMSC+RLM pipeline in fairseq-signals using Hydra configuration. The constituent parts are described below:
- fairseq-hydra-train: Entry point for training with Hydra configs in fairseq-signals.
- task.data=/path/to/output: Path to your preprocessed dataset (e.g., ECG signals prepared for classification).
- model.num_labels=$num_labels: Number of output classes for classification. Replace $num_labels with the actual integer (83 for ptb-xl version and 164 for mimic-iv-ecg version).
- model.model_path=/path/to/checkpoint.pt: Path to the pretrained checkpoint you’re fine‑tuning (e.g., a wav2vec model trained on ECG). For this purpose, a pretrained checkpoint named 'mimic_iv_ecg_physionet_pretrained.pt' extracted from Hugging Face (https://huggingface.co/wanglab/ecg-fm) has been used. Even though the naming convention indicates the checkpoint has been trained using mimic_iv_ecg data, I have successfully been able to use it for PTB-XL data too, as I could not find any ready-made solution built for PTB-XL dataset till date.
- --config-dir ...:Points Hydra to the directory containing YAML configs for this experiment. Here it’s the ECG transformer grounding classification configs.
- --config-name base_total: Selects the specific config file (base_total.yaml) from that directory. This defines hyperparameters like optimizer, learning rate, batch size, etc.

* What W2V+CMSC+RLM means:
- W2V → wav2vec backbone (self‑supervised ECG representation).
- CMSC → Contrastive Multiscale Clustering (a pretraining/fusion strategy).
- RLM → Representation Learning Module (extra grounding for classification).
Together, this pipeline fine‑tunes a pretrained ECG encoder with classification grounding.

4. For Resnet + Attention model:
```shell script
      $ fairseq-hydra-train task.data=/path/to/output \
            model.num_labels=$num_labels \
            --config-dir /fairseq-signals/examples/scratch/ecg_classification/resnet \
            --config-name nejedly2021_total
```
*Notes:*
* *Resnet + Attention* model fine‑tunes or trains a ResNet ECG classifier on your dataset, following the experimental setup described in Nejedly’s 2021 paper. It is part of the scratch examples in fairseq-signals, meaning it is a baseline training recipe rather than a pretrained model fine‑tuning. The constituent parts are described below:
- fairseq-hydra-train: Entry point for training models in fairseq-signals using Hydra configuration.
- task.data=/path/to/output: Path to your preprocessed ECG dataset (the directory containing training/validation/test splits).
- model.num_labels=$num_labels: Number of output classes for classification (83 for ptb-xl version and 164 for mimic-iv-ecg version).
- --config-dir ...resnet: Points Hydra to the configuration directory for ResNet‑based ECG classification experiments.
- --config-name nejedly2021_total: Selects the specific YAML config file (nejedly2021_total.yaml) that defines the training setup used in Nejedly et al. (2021). This config typically specifies:
      - ResNet architecture parameters
      - Optimizer and learning rate schedule
      - Training epochs and batch size
      - Evaluation metrics

5. For SE-WRN model:
```shell script
      $ fairseq-hydra-train task.data=/path/to/output \
            model.num_labels=$num_labels \
            --config-dir /fairseq-signals/examples/scratch/ecg_classification/resnet \
            --config-name se_wrn_total
```
*Notes:*
* The above command is another Hydra‑based training recipe in fairseq‑signals, this time using a Squeeze‑and‑Excitation Wide Residual Network (SE‑WRN) for ECG classification. 
- fairseq-hydra-train: The training launcher that uses Hydra configs.
- task.data=/path/to/output: Path to your preprocessed ECG dataset (train/valid/test splits).
- model.num_labels=$num_labels: Number of diagnostic categories you want to classify.(83 for ptb-xl version and 164 for mimic-iv-ecg version).
- --config-dir ...resnet: Points Hydra to the directory containing ResNet‑style ECG classification configs.
- --config-name se_wrn_total: Selects the YAML config file for the SE‑WRN model.
      - SE: Squeeze‑and‑Excitation blocks, which adaptively recalibrate channel weights to emphasize informative features.
      - WRN: Wide Residual Network, a ResNet variant with wider layers for better capacity.
      - total: typically indicates the full training setup (optimizer, scheduler, augmentation, etc.).

# Instructions to run LLM Modeling steps
1. Same mapping step mentioned as step 1 in the above 'Run QA Experiments' section.

2. Same preprocessing step mentioned as step 2 in the above 'Run QA Experiments' section.

3. Sample 10% from the test set.
```shell script
      $ python llm_modeling/random_sample.py /path/to/output \
            --subset test \
```

* *Notes:*
  - It will sample 10% from test.tsv in the /path/to/output/ directory and output the sampled manifest file test_sampled.tsv in the same directory.
  - /path/to/output: This is the root directory which contains the preprocessed .mat files along with .TSV files generated after the preprocessing step.

4. Run the experiments:
```shell script
      $ python llm_modeling/llm_modeling.py \
          +openai_model=$model_name \
          +openai_api_key=$api_key \
          +ptbxl_or_mimic_iv=ptbxl \
          common_eval.path=/path/to/checkpoint.pt \
          task.data=/path/to/output \
          dataset.valid_subset=test_sampled \
          --config-dir llm_modeling/config \
          --config-name infer_llm
```

* This command is running inference with an LLM‑augmented ECG model using the llm_modeling.py script. Uses the same pretrained model checkpoint as discussed for upperbound experiments. 
- python llm_modeling/llm_modeling.py
Launches the custom script that integrates ECG encoders with large language models (LLMs).
- +openai_model=$model_name
Specifies which OpenAI model to use (e.g., gpt-4.1 etc.).
- +openai_api_key=$api_key
Passes your OpenAI API key so the script can query the LLM.
- +ptbxl_or_mimic_iv=ptbxl
Chooses the dataset source. Here, it’s set to PTB‑XL (you could switch to mimic_iv for MIMIC‑IV ECG).
- common_eval.path=/path/to/checkpoint.pt
Path to the pretrained ECG encoder checkpoint you want to evaluate with the LLM.
- task.data=/path/to/output
Directory containing your preprocessed ECG dataset.
- dataset.valid_subset=test_sampled
Indicates which subset of the dataset to run inference on (here, a sampled test set).
- --config-dir llm_modeling/config
Points Hydra to the configuration directory for LLM modeling.
- --config-name infer_llm
Selects the specific YAML config (infer_llm.yaml) that defines inference settings (batch size, fusion module, evaluation metrics, etc.).

