import os
import time
import json
import urllib.request
import torch
import torch.nn.functional as F
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from tqdm import tqdm

# Directory and URLs for HellaSwag dataset
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")
hellaswags = {
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
}

def download_file(url, filename):
    """Downloads a file from a URL to a local path."""
    with urllib.request.urlopen(url) as response, open(filename, 'wb') as out_file:
        data = response.read()
        out_file.write(data)

def download(split):
    """Downloads HellaSwag dataset to DATA_CACHE_DIR."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

def render_example(example, tokenizer):
    """Renders the example as tokens, mask, and label for model input."""
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    ctx_tokens = tokenizer.encode(ctx, add_special_tokens=False)
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = tokenizer.encode(end, add_special_tokens=False)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))

    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return tokens, mask, label

def iterate_examples(split):
    """Yields examples from the HellaSwag dataset."""
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example

@torch.no_grad()
def benchmark_hellaswag(model, tokenizer, device):
    """Benchmarks the GPT-Neo 125M model on the HellaSwag validation set."""
    model.to(device)
    model.eval()
    num_correct = 0
    num_total = 0

    for example in iterate_examples("val"):
        tokens, mask, label = render_example(example, tokenizer)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # Get logits
        logits = model(tokens).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_tokens = tokens[..., 1:].contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)
        shift_mask = mask[..., 1:].contiguous()
        masked_shift_losses = shift_losses * shift_mask
        sum_loss = masked_shift_losses.sum(dim=1)
        pred = sum_loss.argmin().item()
        num_correct += int(pred == label)
        num_total += 1

    accuracy = num_correct / num_total * 100
    return accuracy

def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Benchmarking
    print("Starting benchmarking...")
    accuracy = benchmark_hellaswag(model, tokenizer, device)
    benchmarking_time = time.time() - start_time

    # Print results
    print(f"Benchmarking completed in {benchmarking_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()

