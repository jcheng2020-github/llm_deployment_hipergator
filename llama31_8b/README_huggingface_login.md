# Hugging Face Login (README)

To download Llama or any other gated Hugging Face model on the cluster, you must authenticate once using a **personal access token (PAT)**. This process is done **outside Slurm**, on a login node.

---

## 1. Create a Hugging Face Token

1. Go to: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token**
3. Choose:

   * **Type:** *Read* (recommended)
   * **Name:** anything you want
4. Copy the generated token
   It looks like:

   ```
   hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

---

## 2. Accept Model License (Required for Llama)

Before downloading Llama models, visit the model page (e.g., Llama-3.1-8B-Instruct):

[https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

Click **“Agree and Access”** to accept the license.
Without this step, downloads will fail even if you are logged in.

---

## 3. Login from the HPC Cluster

On a **login node** (not inside an sbatch job):

```bash
module load anaconda
conda activate llama-b200

huggingface-cli login
```

Paste your token when prompted.

If successful you will see:

```
Token is valid.
Login successful.
```

Your token is stored in:

```
~/.huggingface/token
```

Slurm jobs will automatically use this for authentication.

---

## 4. Test Login (optional)

Run:

```bash
huggingface-cli whoami
```

You should see your Hugging Face username.

---

## 5. Using Hugging Face Inside Slurm Jobs

Inside sbatch scripts, you don’t need to run `huggingface-cli login`.
Instead, you securely store your token once and let jobs read it.

### 5.1 Store the token securely (one-time, on a login node)

```bash
mkdir -p ~/.secrets
chmod 700 ~/.secrets

# Replace hf_xxx... with your actual token
echo "hf_xxxxxxxxxxxxxxxxxxxxx" > ~/.secrets/hf_token

chmod 600 ~/.secrets/hf_token
```

### 5.2 Use the token in sbatch scripts

In your `*.sbatch` file, before running Python:

```bash
# Load Hugging Face token from a protected file
export HF_TOKEN="$(cat ~/.secrets/hf_token)"

# (optional) put caches inside working folder
export HF_HOME="$PWD/.hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_CACHE="$HF_HOME/hub"

mkdir -p "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE"
```

The `huggingface_hub` and `transformers` libraries will automatically pick up `HF_TOKEN` and authenticate to gated models (like Llama).

You still must have **accepted the model license** on the Hugging Face website for the specific model you want (e.g. Llama-3.1-8B-Instruct).

---

## Summary

| Step | What You Do                                      |
| ---- | ------------------------------------------------ |
| 1    | Create a Hugging Face access token               |
| 2    | Accept the Llama model license on Hugging Face   |
| 3    | Save the token securely in `~/.secrets/hf_token` |
| 4    | In sbatch, set `HF_TOKEN` from that file         |
| 5    | Run sbatch jobs normally                         |

Once this is set up, Slurm jobs can download and use gated models (like Llama) non-interactively and without exposing the token in code or logs.
