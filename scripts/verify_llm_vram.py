"""
Pre-check: confirm Qwen3-4B QLoRA fine-tuning fits in 4GB VRAM before
committing to a real run. Runs one forward+backward step on a batch
shaped like the actual extraction task and reports peak memory.
"""

import torch
from unsloth import FastLanguageModel

MAX_SEQ_LENGTH = 512  # matches the task: short announcements, not long documents

print("Loading Qwen3-4B in 4-bit...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-bnb-4bit",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Shaped like the real task: a short announcement in, a structured record out
sample_prompt = (
    "Extract the structured event from this NSE announcement:\n\n"
    "Tata Motors Limited has received an order win from the Indian Railways "
    "for supply of 500 electric buses, valued at approximately Rs. 850 crore, "
    "to be delivered over the next 18 months.\n\nJSON:"
)
sample_target = (
    '{"entity": "TATAMOTORS", "event_type": "order_win", "polarity": 1, '
    '"magnitude": "material", "horizon": "medium", "confidence": 0.85}'
)

inputs = tokenizer(
    sample_prompt + " " + sample_target,
    return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH,
).to("cuda")

torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

model.train()
outputs = model(**inputs, labels=inputs["input_ids"])
outputs.loss.backward()

peak_gb = torch.cuda.max_memory_allocated() / 1e9
total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

print(f"\nLoss: {outputs.loss.item():.4f}")
print(f"Peak VRAM used: {peak_gb:.2f} GB")
print(f"Card total VRAM: {total_gb:.2f} GB")
print(f"Headroom: {total_gb - peak_gb:.2f} GB")

if peak_gb < total_gb * 0.85:
    print("\nPASS — fits with reasonable margin.")
elif peak_gb < total_gb:
    print("\nTIGHT — fits, but little margin. If the real run OOMs, "
          "drop to r=4 or shorten max_seq_length.")
else:
    print("\nFAIL — exceeds available VRAM. Fall back to Qwen3-1.7B.")