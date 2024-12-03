# LLM-Reasoning-and-Correction

This is a step-by-step process to execute this code. Please make sure the config.py file is up to date.

Steps to execute the code:

1. Create necessary directories.

```bash
mkdir data base_models saved_models
```

2. Download the MATH dataset and store it in the data folder.

```bash
cd data
wget https://people.eecs.berkeley.edu/~hendrycks/MATH.tar
tar -xvf MATH.tar
cd ..
```

3. Create conda environment using environment.yml.

```bash
conda env create -f environment.yml
```

4. Activate the environment.

```bash
conda activate llm-self-correct
```

5. Create folder base_models and download the base model from Huggingface by running main.py in download task. Please note a token might be required to access certain models.

```bash
cd code/
python main.py --task download
```

6. Once downloaded, run the main.py in train task. 

```bash
python main.py --task train
```

Or, use the run.slurm file for submitting a job in HPRC.

```bash
cd ..
sbatch run.slurm
```

7. To run the saved model in evaluation mode, use the following command.

```bash
python main.py --task evaluate
```