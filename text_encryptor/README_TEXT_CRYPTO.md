# Neural Text Encryption Tool

> **⚠️ RESEARCH DEMONSTRATION ONLY** — This is a proof-of-concept implementation of
> adversarial neural cryptography (Abadi & Andersen, 2016). It uses 16-bit keys which
> are trivially brute-forceable. Do **not** use this for anything requiring real security.

---

## Overview

Four files implement neural text encryption on top of trained Alice/Bob models from
`main_enhanced.py`:

| File | Purpose |
|---|---|
| `text_encryptor.py` | Core `TextEncryptor` class — encrypt/decrypt text |
| `text_crypto_tool.py` | CLI interface (encrypt, decrypt, demo, evaluate, batch) |
| `demo_text.py` | End-to-end visual demo with figures |
| `text_security_eval.py` | Multi-run Eve robustness evaluation |

---

## Installation

Same dependencies as `main_enhanced.py`:

```bash
pip install tensorflow>=2.10 numpy>=1.21 matplotlib
```

---

## Quick Start

### Encrypt a message
```bash
python text_crypto_tool.py encrypt \
  --message "Hello World" \
  --output encrypted.npz \
  --save-key key.txt
```

### Decrypt
```bash
python text_crypto_tool.py decrypt \
  --input encrypted.npz \
  --key-file key.txt
```

### Interactive demo
```bash
python text_crypto_tool.py demo --mode ascii
# or
python demo_text.py --message "Your secret message"
```

### Security evaluation
```bash
python text_crypto_tool.py evaluate --message "Secret" --steps 5000
# or
python text_security_eval.py --message "Secret" --runs 5 --steps 5000
```

### Encrypt/decrypt a text file
```bash
python text_crypto_tool.py encrypt --input report.txt --output report_enc.npz --save-key k.txt
python text_crypto_tool.py decrypt --input report_enc.npz --key-file k.txt --output report_dec.txt
```

### Batch operations
```bash
python text_crypto_tool.py batch-encrypt \
  --input-dir ./docs --output-dir ./encrypted --save-key master.key

python text_crypto_tool.py batch-decrypt \
  --input-dir ./encrypted --output-dir ./decrypted --key-file master.key
```

---

## Encoding Modes

### `ascii` mode (recommended for text)
- **8 bits per character** → 2 chars per 16-bit block
- Works with printable ASCII (codes 32–126)
- Faster and more human-readable output
- **Limitation**: bit 7 (MSB) is always `−1` for printable ASCII, introducing
  a predictable bias that slightly weakens post-training robustness
  (typical result: ~2/5 secure runs vs 5/5 for random mode)

### `random` mode (maximum security)
- **16 bits per character** → 1 char per 16-bit block
- Works for any text including extended characters
- No structural bias — uniform distribution matches training data
- More robust post-training security (5/5 secure runs with seed 13)

---

## API Reference

### `TextEncryptor`

```python
from text_encryptor import TextEncryptor

enc = TextEncryptor(
    checkpoint_dir="./checkpoints/ascii_quadratic",  # or seed_13 for random
    mode="ascii",    # "ascii" or "random"
    msg_size=16,
    key_size=16,
)

# Generate a key
key = enc.generate_key()                    # returns (16,) float32 array of ±1

# Save / load key
enc.save_key(key, "key.txt")
key = enc.load_key("key.txt")

# Encrypt
result = enc.encrypt_text("Hello World", key=key, output_path="enc.npz")
# result["ciphertext"] shape: (n_blocks, 16) float32
# result["key"], result["original_length"], result["mode"]

# Decrypt
plaintext = enc.decrypt_text("enc.npz", key)             # from file
plaintext = enc.decrypt_text(result["ciphertext"], key,  # from array
                              original_length=len("Hello World"))

# File operations
enc.encrypt_file("report.txt", "report_enc.npz", key=key)
enc.decrypt_file("report_enc.npz", "report_dec.txt", key=key)

# Quick security check (trains a fresh Eve)
sec = enc.quick_security_check("Hello World", key, eve_steps=1000)
print(sec["best_err"], sec["secure"])
```

---

## Key Format

Keys are saved as plain comma-separated `+1`/`-1` values:

```
1,-1,1,-1,1,1,-1,-1,1,-1,1,1,-1,1,-1,1
```

Load with:
```python
key = TextEncryptor.load_key("key.txt")
```

---

## Encrypted File Format (`.npz`)

```python
data = np.load("encrypted.npz", allow_pickle=True)
data["ciphertext"]       # (n_blocks, 16) float32 — Alice's output
data["key"]              # (16,) float32 — the key used
data["original_length"]  # int — original character count (for trimming padding)
data["mode"]             # str — "ascii" or "random"
data["msg_size"]         # int — 16
```

---

## Security Limitations

| Property | Neural Cipher | AES-128 |
|---|---|---|
| Key size | 16 bits (weak!) | 128 bits |
| Block size | 16 bits | 128 bits |
| Formal security proof | ❌ None | ✅ IND-CPA |
| Brute-force resistance | ❌ 65,536 keys | ✅ 2¹²⁸ keys |
| Speed | Slow (neural inference) | Very fast (hardware) |
| Integrity protection | ❌ None | ✅ (AES-GCM) |

**ASCII mode specific weakness:** Printable ASCII characters always have
bit 7 = 0, creating a predictable bit in every block. A sophisticated
attacker can exploit this. Random mode avoids this bias.

**Do not use for**: passwords, private communications, financial data,
legal documents, or any security-sensitive purpose.

---

## Checkpoints Required

| Mode | Checkpoint directory | Trained with |
|---|---|---|
| `ascii` | `./checkpoints/ascii_quadratic/` | `--mode ascii --loss_fn quadratic` |
| `random` | `./checkpoints/seed_13/` | `--mode random --seed 13` |

Each directory must contain `alice_final.weights.h5` and `bob_final.weights.h5`.

If checkpoints are missing, the tool will warn and use random weights
(output will be garbage).

---

## Reference

Abadi, M. & Andersen, D.G. (2016). *Learning to Protect Communications with
Neural Cryptography*. arXiv:1610.06918.
