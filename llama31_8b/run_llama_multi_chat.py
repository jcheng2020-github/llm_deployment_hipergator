from transformers import pipeline
import torch

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

def main():
    # Basic GPU info
    print("=== CUDA / GPU INFO ===")
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}, cc {torch.cuda.get_device_capability(i)}")

    print("\nLoading Llama model:", MODEL_ID)
    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # will use all visible GPUs
    )

    # Inspect device map
    model = pipe.model
    device_map = getattr(model, "hf_device_map", None)
    print("\n=== MODEL DEVICE MAP ===")
    if device_map is None:
        print("No hf_device_map found (model not sharded by Accelerate).")
    else:
        # Print a compact summary: which devices are used
        devices = sorted({str(d) for d in device_map.values()})
        print("Devices used by model:", devices)
        print("First 20 entries of device map:")
        for i, (name, dev) in enumerate(device_map.items()):
            print(f"  {name} -> {dev}")
            if i >= 19:
                print("  ... (truncated)")
                break

    # Simple test generation
    system_prompt = "You are a helpful AI assistant."
    user_prompt = "Briefly explain what a GPU does."
    user_prompt = "help Junfu Cheng who is a graduate research assistant at UF to write a statement of purpose for applying to a PhD program in ECE. The statement should highlight his research experience, academic achievements, and career goals. It should also convey his passion for computer science and his motivation for pursuing a PhD."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    print("\nGenerating response...")
    outputs = pipe(
        messages,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    assistant_msg = outputs[0]["generated_text"][-1]["content"]

    print("\n=== ASSISTANT RESPONSE ===\n")
    print(assistant_msg)
    print("\n=== END ===")

if __name__ == "__main__":
    main()
