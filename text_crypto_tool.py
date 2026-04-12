"""
=============================================================================
  text_crypto_tool.py — Command-Line Interface for Neural Text Encryption
=============================================================================
  Usage examples:
    python text_crypto_tool.py encrypt --message "Hello World" --output enc.npz --save-key key.txt
    python text_crypto_tool.py encrypt --input msg.txt --output enc.npz --save-key key.txt
    python text_crypto_tool.py decrypt --input enc.npz --key-file key.txt
    python text_crypto_tool.py decrypt --input enc.npz --key "1,-1,1,-1,..." --output out.txt
    python text_crypto_tool.py demo --mode ascii
    python text_crypto_tool.py evaluate --message "Secret" --steps 5000
    python text_crypto_tool.py batch-encrypt --input-dir ./docs --output-dir ./enc --key-file k.txt
    python text_crypto_tool.py batch-decrypt --input-dir ./enc  --output-dir ./dec --key-file k.txt
=============================================================================
"""

import argparse
import os
import sys
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_key_string(s: str) -> np.ndarray:
    try:
        arr = np.array([float(x.strip()) for x in s.split(",")], dtype=np.float32)
        if not np.all(np.isin(arr, [-1.0, 1.0])):
            raise ValueError
        return arr
    except Exception:
        print("[ERROR] Key must be comma-separated +1/-1 values, e.g. '1,-1,1,-1,...'")
        sys.exit(1)


def resolve_key(args, enc=None) -> np.ndarray:
    from text_encryptor import TextEncryptor
    key_size = getattr(args, "key_size", 16)

    if hasattr(args, "key") and args.key:
        return parse_key_string(args.key)
    elif hasattr(args, "key_file") and args.key_file and os.path.exists(args.key_file):
        return TextEncryptor.load_key(args.key_file)
    else:
        key = TextEncryptor.generate_key(key_size)
        print(f"[Key] Generated random key: {key.astype(int).tolist()}")
        return key


def get_encryptor(args):
    from text_encryptor import TextEncryptor
    ckpt = getattr(args, "checkpoint_dir", "./checkpoints/seed_13")
    mode = getattr(args, "mode", "random")
    return TextEncryptor(checkpoint_dir=ckpt, mode=mode)


# ── Subcommand handlers ───────────────────────────────────────────────────────

def cmd_encrypt(args):
    enc = get_encryptor(args)
    key = resolve_key(args)

    # Get plaintext from --message or --input file
    if getattr(args, "message", None):
        plaintext = args.message
    elif getattr(args, "input", None):
        with open(args.input, "r", encoding="utf-8") as f:
            plaintext = f.read()
        print(f"[Read] {args.input}  ({len(plaintext)} chars)")
    else:
        print("[ERROR] Provide --message or --input"); sys.exit(1)

    output = getattr(args, "output", None) or "encrypted.npz"
    result = enc.encrypt_text(plaintext, key=key, output_path=output)

    if getattr(args, "save_key", None):
        enc.save_key(result["key"], args.save_key)

    print(f"\n[Done] Encrypted → {output}")
    print(f"       Key: {result['key'].astype(int).tolist()}")


def cmd_decrypt(args):
    enc = get_encryptor(args)
    key = resolve_key(args)

    input_path = getattr(args, "input", None)
    if not input_path or not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}"); sys.exit(1)

    output = getattr(args, "output", None)
    plaintext = enc.decrypt_text(input_path, key, output_path=output)

    if output is None:
        print(f"\n[Decrypted]\n{plaintext}")


def cmd_batch_encrypt(args):
    enc = get_encryptor(args)
    key = resolve_key(args)

    if getattr(args, "save_key", None):
        enc.save_key(key, args.save_key)

    os.makedirs(args.output_dir, exist_ok=True)
    exts = {".txt", ".md", ".csv", ".log", ".py", ".json"}
    files = [
        f for f in os.listdir(args.input_dir)
        if os.path.splitext(f.lower())[1] in exts
    ]
    if not files:
        print(f"[Batch] No text files found in {args.input_dir}"); return

    print(f"[Batch] Encrypting {len(files)} files …")
    for fname in sorted(files):
        src = os.path.join(args.input_dir, fname)
        dst = os.path.join(args.output_dir, os.path.splitext(fname)[0] + "_enc.npz")
        try:
            enc.encrypt_file(src, dst, key=key)
        except Exception as e:
            print(f"[Batch] ERROR on {fname}: {e}")


def cmd_batch_decrypt(args):
    enc = get_encryptor(args)
    key = resolve_key(args)

    os.makedirs(args.output_dir, exist_ok=True)
    files = [f for f in os.listdir(args.input_dir) if f.endswith(".npz")]
    if not files:
        print(f"[Batch] No .npz files found in {args.input_dir}"); return

    print(f"[Batch] Decrypting {len(files)} files …")
    for fname in sorted(files):
        src = os.path.join(args.input_dir, fname)
        stem = fname.replace("_enc", "").replace(".npz", "")
        dst = os.path.join(args.output_dir, stem + "_dec.txt")
        try:
            enc.decrypt_file(src, dst, key=key)
        except Exception as e:
            print(f"[Batch] ERROR on {fname}: {e}")


def cmd_demo(args):
    from text_encryptor import TextEncryptor
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mode = getattr(args, "mode", "ascii")
    ckpt_map = {
        "ascii":  "./checkpoints/ascii_quadratic",
        "random": "./checkpoints/seed_13",
    }
    ckpt = ckpt_map.get(mode, "./checkpoints/seed_13")

    print("\n" + "=" * 62)
    print("  NEURAL TEXT ENCRYPTION DEMO")
    print(f"  Mode: {mode.upper()}  |  Checkpoint: {ckpt}")
    print("  [RESEARCH DEMONSTRATION — NOT production security]")
    print("=" * 62)

    enc = TextEncryptor(checkpoint_dir=ckpt, mode=mode)
    key = enc.generate_key()
    print(f"\nKey: {key.astype(int).tolist()}\n")

    messages = [
        "Hello World!",
        "Neural cryptography is fascinating.",
        "The quick brown fox jumps over the lazy dog.",
        "Secret: password123",
    ]

    results = []
    for msg in messages:
        result   = enc.encrypt_text(msg, key=key)
        decrypted = enc.decrypt_text(result["ciphertext"], key, original_length=len(msg))
        match = (decrypted == msg)

        # Wrong-key decrypt
        wrong_key = enc.generate_key()
        if np.all(wrong_key == key):
            wrong_key[0] *= -1
        dec_wrong = enc.decrypt_text(result["ciphertext"], wrong_key, original_length=len(msg))

        print(f"\n  Original  : '{msg}'")
        print(f"  Cipher[0] : {result['ciphertext'][0, :6].round(3).tolist()} ...")
        print(f"  Decrypted : '{decrypted}' {'✅' if match else '❌'}")
        print(f"  Wrong key : '{dec_wrong}'")
        results.append((msg, decrypted, dec_wrong, match))

    # Quick security check on last message
    print(f"\n[Demo] Running quick security check (500 Eve steps) …")
    sec = enc.quick_security_check(messages[-1], key, eve_steps=500)
    status = "✅ SECURE" if sec["secure"] else "⚠️  PARTIAL BREAK"
    print(f"  Eve best error : {sec['best_err']:.3f} / {enc.msg_size}")
    print(f"  Random baseline: {sec['random_baseline']:.1f}")
    print(f"  Status         : {status}")

    # Save simple figure
    out_fig = getattr(args, "output", "text_demo.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    rows = [["Original", "Decrypted", "Match", "Wrong-key (first 12)"]]
    for msg, dec, wrong, ok in results:
        rows.append([
            msg[:30],
            dec[:30],
            "✓" if ok else "✗",
            wrong[:12],
        ])
    table = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(4)))
    ax.set_title(
        f"Neural Text Encryption Demo — {mode.upper()} mode\n"
        f"[RESEARCH DEMO — NOT production security]",
        fontsize=11, color="darkred",
    )
    plt.tight_layout()
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Demo] Figure saved → {out_fig}")
    print("=" * 62)


def cmd_evaluate(args):
    from text_encryptor import TextEncryptor
    from main_enhanced import AttackerNet, l1_distance, eve_loss
    from tensorflow import keras
    import tensorflow as tf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mode = getattr(args, "mode", "ascii")
    ckpt_map = {
        "ascii":  "./checkpoints/ascii_quadratic",
        "random": "./checkpoints/seed_13",
    }
    ckpt = ckpt_map.get(mode, "./checkpoints/seed_13")
    steps = getattr(args, "steps", 5000)

    msg = getattr(args, "message", None)
    if not msg and getattr(args, "input", None):
        with open(args.input) as f:
            msg = f.read()
    if not msg:
        msg = "The quick brown fox jumps over the lazy dog."

    enc = TextEncryptor(checkpoint_dir=ckpt, mode=mode)
    key = enc.generate_key()
    result = enc.encrypt_text(msg, key=key)
    blocks = enc._text_to_blocks(msg)
    cipher = result["ciphertext"]
    msg_size = enc.msg_size

    print(f"\n[Eval] Training fresh Eve for {steps} steps …")
    fresh_eve = AttackerNet(msg_size, False, name="eval_eve")
    opt_eve   = keras.optimizers.Adam(0.0008)
    _ = fresh_eve(tf.zeros([1, msg_size]))

    best_err  = float("inf")
    curve     = []
    BATCH     = min(512, len(cipher))

    for step in range(steps):
        idx   = np.random.randint(0, len(cipher), BATCH)
        b_c   = tf.constant(cipher[idx])
        b_msg = tf.constant(blocks[idx])
        with tf.GradientTape() as tape:
            dec  = fresh_eve(b_c)
            loss = eve_loss(b_msg, dec)
        grads = tape.gradient(loss, fresh_eve.trainable_variables)
        opt_eve.apply_gradients(zip(grads, fresh_eve.trainable_variables))
        err = float(tf.reduce_mean(l1_distance(b_msg, dec)).numpy())
        if err < best_err:
            best_err = err
        if step % max(steps // 20, 1) == 0:
            curve.append((step, err))
            print(f"  Step {step:5d}/{steps}  Eve error: {err:.3f}/{msg_size}")

    random_base = msg_size / 2.0
    secure = best_err >= random_base * 0.85
    print(f"\n  Best Eve error : {best_err:.3f} / {msg_size}")
    print(f"  Random baseline: {random_base:.1f}")
    print(f"  Status         : {'✅ SECURE' if secure else '⚠️  PARTIAL BREAK'}")

    # Plot
    xs = [c[0] for c in curve]; ys = [c[1] for c in curve]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, color="#E74C3C", linewidth=2, label="Eve reconstruction error")
    ax.axhline(random_base, color="#95A5A6", linestyle="--",
               label=f"Random baseline ({random_base} bits)")
    ax.set_xlabel("Step"); ax.set_ylabel("Error (bits)")
    ax.set_title(f"Security Evaluation — '{msg[:40]}...'\nMode: {mode}")
    ax.legend(); ax.grid(True, alpha=0.3)
    out = getattr(args, "output", "text_security_eval.png")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"[Eval] Plot → {out}")


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        prog="text_crypto_tool",
        description="Neural text encryption CLI (Abadi & Andersen 2016)"
    )
    p.add_argument("--checkpoint-dir", default=None, dest="checkpoint_dir",
                   help="Override checkpoint directory")
    p.add_argument("--mode", default="ascii", choices=["ascii", "random"],
                   help="Encoding mode (default: ascii)")
    sub = p.add_subparsers(dest="command", required=True)

    # encrypt
    e = sub.add_parser("encrypt", help="Encrypt a text message or file")
    e.add_argument("--message", default=None, help="Plaintext message string")
    e.add_argument("--input",   default=None, help="Input text file path")
    e.add_argument("--output",  default="encrypted.npz")
    e.add_argument("--key",      default=None, help="Key as '1,-1,...'")
    e.add_argument("--key-file", default=None, dest="key_file")
    e.add_argument("--save-key", default=None, dest="save_key")

    # decrypt
    d = sub.add_parser("decrypt", help="Decrypt an encrypted .npz file")
    d.add_argument("--input",   required=True)
    d.add_argument("--output",  default=None)
    d.add_argument("--key",      default=None)
    d.add_argument("--key-file", default=None, dest="key_file")

    # demo
    dm = sub.add_parser("demo", help="Interactive demo with sample messages")
    dm.add_argument("--output", default="text_demo.png")

    # evaluate
    ev = sub.add_parser("evaluate", help="Security evaluation (Eve retraining)")
    ev.add_argument("--message", default=None)
    ev.add_argument("--input",   default=None)
    ev.add_argument("--steps",   type=int, default=5000)
    ev.add_argument("--output",  default="text_security_eval.png")

    # batch-encrypt
    be = sub.add_parser("batch-encrypt", help="Encrypt all text files in a directory")
    be.add_argument("--input-dir",  required=True, dest="input_dir")
    be.add_argument("--output-dir", required=True, dest="output_dir")
    be.add_argument("--key",        default=None)
    be.add_argument("--key-file",   default=None, dest="key_file")
    be.add_argument("--save-key",   default=None, dest="save_key")

    # batch-decrypt
    bd = sub.add_parser("batch-decrypt", help="Decrypt all .npz files in a directory")
    bd.add_argument("--input-dir",  required=True, dest="input_dir")
    bd.add_argument("--output-dir", required=True, dest="output_dir")
    bd.add_argument("--key",        default=None)
    bd.add_argument("--key-file",   default=None, dest="key_file")

    return p


HANDLERS = {
    "encrypt":       cmd_encrypt,
    "decrypt":       cmd_decrypt,
    "demo":          cmd_demo,
    "evaluate":      cmd_evaluate,
    "batch-encrypt": cmd_batch_encrypt,
    "batch-decrypt": cmd_batch_decrypt,
}

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    # Apply checkpoint-dir override to mode-specific defaults
    if args.checkpoint_dir:
        pass  # already set; get_encryptor() will use it
    else:
        ckpt_map = {
            "ascii":  "./checkpoints/ascii_quadratic",
            "random": "./checkpoints/seed_13",
        }
        args.checkpoint_dir = ckpt_map.get(args.mode, "./checkpoints/seed_13")

    handler = HANDLERS.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
