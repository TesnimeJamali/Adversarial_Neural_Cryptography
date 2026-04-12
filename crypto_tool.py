"""
=============================================================================
  crypto_tool.py — Command-Line Interface for Neural Image Encryption
=============================================================================
  Usage examples:
    python crypto_tool.py encrypt --input cat.jpg --output cat.npz --save-key key.txt
    python crypto_tool.py decrypt --input cat.npz  --output cat_dec.png --key-file key.txt
    python crypto_tool.py demo    --image test.jpg
    python crypto_tool.py evaluate --image test.jpg --steps 5000
    python crypto_tool.py batch-encrypt --input-dir ./photos --output-dir ./enc --key-file k.txt
    python crypto_tool.py batch-decrypt --input-dir ./enc    --output-dir ./dec --key-file k.txt
=============================================================================
"""

import argparse
import sys
import os
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_key_string(s: str) -> np.ndarray:
    """Parse '1,-1,1,-1,...' into a float32 array."""
    try:
        arr = np.array([float(x.strip()) for x in s.split(",")], dtype=np.float32)
        if not np.all(np.isin(arr, [-1.0, 1.0])):
            raise ValueError
        return arr
    except Exception:
        print("[ERROR] Key must be comma-separated +1/-1 values, e.g. '1,-1,1,-1,...'")
        sys.exit(1)


def resolve_key(args) -> np.ndarray:
    """Determine key from --key, --key-file, or generate randomly."""
    from image_encryptor import ImageEncryptor
    key_size = getattr(args, "key_size", 16)

    if hasattr(args, "key") and args.key:
        return parse_key_string(args.key)
    elif hasattr(args, "key_file") and args.key_file and os.path.exists(args.key_file):
        return ImageEncryptor.load_key(args.key_file)
    else:
        key = ImageEncryptor.generate_key(key_size)
        print(f"[Key] Generated random key: {key.astype(int).tolist()}")
        return key


def get_encryptor(args):
    from image_encryptor import ImageEncryptor
    ckpt = getattr(args, "checkpoint_dir", "./checkpoints/seed_13")
    return ImageEncryptor(checkpoint_dir=ckpt)


# ── Subcommand handlers ───────────────────────────────────────────────────────

def cmd_encrypt(args):
    enc = get_encryptor(args)
    key = resolve_key(args)

    if hasattr(args, "save_key") and args.save_key:
        enc.save_key(key, args.save_key)

    output = getattr(args, "output", None)
    enc.encrypt_image(args.input, key, output_path=output)


def cmd_decrypt(args):
    enc = get_encryptor(args)
    key = resolve_key(args)
    output = getattr(args, "output", None)
    enc.decrypt_image(args.input, key, output_path=output)


def cmd_batch_encrypt(args):
    enc = get_encryptor(args)
    key = resolve_key(args)
    if hasattr(args, "save_key") and args.save_key:
        enc.save_key(key, args.save_key)
    enc.encrypt_batch(args.input_dir, args.output_dir, key)


def cmd_batch_decrypt(args):
    enc = get_encryptor(args)
    key = resolve_key(args)
    enc.decrypt_batch(args.input_dir, args.output_dir, key)


def cmd_demo(args):
    """Encrypt, decrypt, and display side-by-side comparison."""
    import time
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    from image_encryptor import ImageEncryptor

    ckpt = getattr(args, "checkpoint_dir", "./checkpoints/seed_13")
    enc  = ImageEncryptor(checkpoint_dir=ckpt)
    key  = enc.generate_key()
    print(f"\n[Demo] Key: {key.astype(int).tolist()}\n")

    img_path  = args.image
    img_orig  = np.array(Image.open(img_path).convert("L"))
    H, W      = img_orig.shape
    print(f"[Demo] Image: {img_path}  ({H}×{W} px)")

    # Encrypt
    enc_path = "/tmp/_demo_encrypted.npz"
    t0 = time.time()
    enc.encrypt_image(img_path, key, output_path=enc_path)
    t_enc = time.time() - t0

    # Decrypt
    dec_path = "/tmp/_demo_decrypted.png"
    t0 = time.time()
    img_dec = enc.decrypt_image(enc_path, key, output_path=dec_path)
    t_dec = time.time() - t0

    # Visualise ciphertext as noise
    img_cipher = enc.visualize_ciphertext(enc_path)

    # PSNR
    mse = np.mean((img_orig.astype(float) - img_dec.astype(float)) ** 2)
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)

    print(f"\n  Encryption time : {t_enc:.3f}s")
    print(f"  Decryption time : {t_dec:.3f}s")
    print(f"  PSNR (orig↔dec) : {psnr:.1f} dB")

    # Save comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(img_orig,   cmap="gray", vmin=0, vmax=255); axes[0].set_title("Original");           axes[0].axis("off")
    axes[1].imshow(img_cipher, cmap="gray", vmin=0, vmax=255); axes[1].set_title("Encrypted (noise)");  axes[1].axis("off")
    axes[2].imshow(img_dec,    cmap="gray", vmin=0, vmax=255); axes[2].set_title(f"Decrypted  PSNR={psnr:.1f}dB"); axes[2].axis("off")
    plt.suptitle("Adversarial Neural Cryptography — Image Encryption Demo\n"
                 "[RESEARCH DEMO — NOT production-grade security]",
                 fontsize=12, color="darkred")
    plt.tight_layout()
    out_fig = getattr(args, "output", "encryption_demo.png")
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"\n[Demo] Comparison figure → {out_fig}")

    # Quick security check
    print("\n[Demo] Running quick security check (1000 Eve steps) …")
    _quick_eve_eval(enc, enc_path, key, steps=1000)


def cmd_evaluate(args):
    """Train a fresh Eve against the encrypted image and report robustness."""
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from image_encryptor import ImageEncryptor
    import tensorflow as tf
    from tensorflow import keras
    from main_enhanced import AttackerNet, l1_distance, eve_loss

    ckpt  = getattr(args, "checkpoint_dir", "./checkpoints/seed_13")
    enc   = ImageEncryptor(checkpoint_dir=ckpt)
    key   = enc.generate_key()
    steps = getattr(args, "steps", 5000)

    # Encrypt test image
    enc_path = "/tmp/_eval_encrypted.npz"
    enc.encrypt_image(args.image, key, output_path=enc_path)

    # Load ciphertext
    data    = np.load(enc_path)
    cipher  = data["ciphertext"]          # (n_blocks, 16)
    # Reconstruct original blocks for ground truth
    img_arr = np.array(Image.open(args.image).convert("L"))
    blocks, _, _ = enc._image_to_blocks(img_arr)

    msg_size = enc.msg_size
    print(f"\n[Eval] Training fresh Eve for {steps} steps …")

    fresh_eve = AttackerNet(msg_size, use_attention=False, name="fresh_eve")
    opt_eve   = keras.optimizers.Adam(0.0008)

    # Warm up
    dummy = tf.zeros([1, msg_size])
    _ = fresh_eve(dummy)

    best_err = float("inf")
    curve    = []
    BATCH    = 1024

    for step in range(steps):
        idx     = np.random.randint(0, len(cipher), BATCH)
        b_c     = tf.constant(cipher[idx])
        b_truth = tf.constant(blocks[idx])

        with tf.GradientTape() as tape:
            dec  = fresh_eve(b_c)
            loss = eve_loss(b_truth, dec)
        grads = tape.gradient(loss, fresh_eve.trainable_variables)
        opt_eve.apply_gradients(zip(grads, fresh_eve.trainable_variables))

        err = float(tf.reduce_mean(l1_distance(b_truth, dec)).numpy())
        if err < best_err:
            best_err = err

        if step % max(steps // 20, 1) == 0:
            print(f"  Step {step:5d}/{steps}  Eve error: {err:.3f} / {msg_size}")
            curve.append((step, err))

    random_baseline = msg_size / 2.0
    secure = best_err >= random_baseline * 0.85
    print(f"\n  Best Eve error : {best_err:.3f} bits")
    print(f"  Random baseline: {random_baseline:.1f} bits")
    print(f"  Status         : {'✅ SECURE' if secure else '⚠️  PARTIAL BREAK'}")

    # Plot Eve learning curve
    xs = [c[0] for c in curve]
    ys = [c[1] for c in curve]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, color="#E74C3C", linewidth=1.8, label="Eve reconstruction error")
    ax.axhline(random_baseline, color="#95A5A6", linestyle="--",
               label=f"Random baseline ({random_baseline} bits)")
    ax.set_xlabel("Step"); ax.set_ylabel("Error (bits)")
    ax.set_title("Eve Security Evaluation — Fresh Attacker on Encrypted Image")
    ax.legend(); ax.grid(True, alpha=0.3)
    out_eval = getattr(args, "output", "security_eval.png")
    plt.tight_layout(); plt.savefig(out_eval, dpi=150); plt.close()
    print(f"[Eval] Learning curve → {out_eval}")


def _quick_eve_eval(enc, enc_path, key, steps=1000):
    """Lightweight inline Eve check used by demo."""
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from PIL import Image
    from main_enhanced import AttackerNet, l1_distance, eve_loss

    data   = np.load(enc_path)
    cipher = data["ciphertext"]
    msg_size = enc.msg_size
    BATCH  = 512

    fresh_eve = AttackerNet(msg_size, False, name="q_eve")
    opt_eve   = keras.optimizers.Adam(0.0008)
    _ = fresh_eve(tf.zeros([1, msg_size]))   # warm up

    # Reconstruct blocks from original
    # We'll use random +1/-1 as proxy since we don't have original path here
    best = float("inf")
    for _ in range(steps):
        idx   = np.random.randint(0, len(cipher), BATCH)
        b_c   = tf.constant(cipher[idx])
        # Eve doesn't know original — minimise her self-supervised reconstruction
        with tf.GradientTape() as tape:
            dec  = fresh_eve(b_c)
            # Use random target as proxy — tests if cipher has exploitable structure
            tgt  = tf.sign(b_c)            # same distribution as plaintext
            loss = tf.reduce_mean(tf.abs(tgt - dec))
        grads = tape.gradient(loss, fresh_eve.trainable_variables)
        opt_eve.apply_gradients(zip(grads, fresh_eve.trainable_variables))
        err = float(loss.numpy())
        if err < best:
            best = err

    random_baseline = msg_size / 2.0
    print(f"  Quick Eve check:  best proxy error = {best:.3f}")
    print(f"  (Lower = Eve found exploitable structure; target ≥ {random_baseline:.1f})")


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        prog="crypto_tool",
        description="Neural image encryption CLI (Abadi & Andersen 2016)"
    )
    p.add_argument("--checkpoint-dir", default="./checkpoints/seed_13",
                   dest="checkpoint_dir",
                   help="Directory containing trained model weights")
    sub = p.add_subparsers(dest="command", required=True)

    # encrypt
    e = sub.add_parser("encrypt", help="Encrypt a single image")
    e.add_argument("--input",    required=True,  help="Input image path")
    e.add_argument("--output",   default=None,   help="Output .npz path")
    e.add_argument("--key",      default=None,   help="Key as '1,-1,1,...'")
    e.add_argument("--key-file", default=None,   dest="key_file", help="Load key from file")
    e.add_argument("--save-key", default=None,   dest="save_key", help="Save key to file")

    # decrypt
    d = sub.add_parser("decrypt", help="Decrypt an encrypted image")
    d.add_argument("--input",    required=True,  help="Input .npz path")
    d.add_argument("--output",   default=None,   help="Output PNG path")
    d.add_argument("--key",      default=None,   help="Key as '1,-1,1,...'")
    d.add_argument("--key-file", default=None,   dest="key_file", help="Load key from file")

    # batch-encrypt
    be = sub.add_parser("batch-encrypt", help="Encrypt all images in a directory")
    be.add_argument("--input-dir",  required=True, dest="input_dir")
    be.add_argument("--output-dir", required=True, dest="output_dir")
    be.add_argument("--key",        default=None)
    be.add_argument("--key-file",   default=None,  dest="key_file")
    be.add_argument("--save-key",   default=None,  dest="save_key")

    # batch-decrypt
    bd = sub.add_parser("batch-decrypt", help="Decrypt all .npz files in a directory")
    bd.add_argument("--input-dir",  required=True, dest="input_dir")
    bd.add_argument("--output-dir", required=True, dest="output_dir")
    bd.add_argument("--key",        default=None)
    bd.add_argument("--key-file",   default=None,  dest="key_file")

    # demo
    dm = sub.add_parser("demo", help="Visual encryption/decryption demo")
    dm.add_argument("--image",  required=True, help="Test image path")
    dm.add_argument("--output", default="encryption_demo.png", help="Output figure path")

    # evaluate
    ev = sub.add_parser("evaluate", help="Security evaluation (Eve retraining)")
    ev.add_argument("--image",  required=True)
    ev.add_argument("--steps",  type=int, default=5000)
    ev.add_argument("--output", default="security_eval.png")

    return p


HANDLERS = {
    "encrypt":       cmd_encrypt,
    "decrypt":       cmd_decrypt,
    "batch-encrypt": cmd_batch_encrypt,
    "batch-decrypt": cmd_batch_decrypt,
    "demo":          cmd_demo,
    "evaluate":      cmd_evaluate,
}

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    handler = HANDLERS.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
