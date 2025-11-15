# llm_deployment_hipergator
Scripts for deploying LLMs on the HiPerGator Slurm cluster at the University of Florida.

This repository contains scripts and environment setups for deploying Large Language Models (LLMs) on the HiPerGator supercomputing cluster at the University of Florida using the Slurm workload manager. It includes examples for environment creation, GPU testing, multi-GPU Llama-3.1-8B inference, and job submission workflows optimized for UF’s HPC environment.

---

## Prerequisites (HiPerGator – University of Florida)

Before using these scripts on HiPerGator, ensure you have:

1. **A HiPerGator account** with compute allocation.
2. **Access to B200 or A100 GPUs** (via SLURM partitions such as `gpu`, `gpu-b200`, etc.).
3. **Conda / Miniconda available** in your environment (`module load conda` or `module load miniconda`).
4. **Hugging Face access token** added to your home directory
   (e.g., `~/.secrets/hf_token` or `~/.huggingface/token`).
5. **Accepted the model license** for Llama-3.1-8B on Hugging Face.
6. **SLURM submission permissions** to run `sbatch`, `srun`, and `salloc`.

---

## Usage Guide

### **1. Create the Conda Environment**

Navigate to the `env_create` directory and run the environment setup script:

```bash
cd env_create
sbatch setup_llama_env.sh
```

This will create a Conda environment named:

```
llama-b200
```

which contains the dependencies needed for running Llama-3.1-8B on B200 GPUs.

---

### **2. Test GPU Availability (Optional but Recommended)**

Confirm that your job can successfully request and use a **B200 GPU**.

From inside `test_b200/`, run:

```bash
cd test_b200
sbatch test_b200.sh
```

This script prints GPU information using `nvidia-smi` and ensures that the B200 GPU is properly allocated to your job.

---

### **3. Run Llama-3.1-8B Inference on B200 GPUs**

After verifying the environment and GPU, run the multi-GPU Llama-3.1-8B job:

```bash
cd llama31_8b
sbatch run_llama_multi_b200.sh
```

This script launches the model and writes inference output to a file named:

```
llama31_8b_multi_%j.out
```

where `%j` is the SLURM job ID.

The Python script `run_llama_multi_chat.py` handles loading the model and generating LLM outputs.

---

* **For further questions, please contact Junfu Cheng at [junfu.cheng@ufl.edu]**
