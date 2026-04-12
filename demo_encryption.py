"""
=============================================================================
  demo_encryption.py — Visual demonstration of neural image encryption
=============================================================================
  Usage:
    python demo_encryption.py --image path/to/image.jpg
    python demo_encryption.py --create-test          # 128×128 synthetic pattern

  Produces a 5-panel figure:
    Row 1: Original | Encrypted (noise) | Decrypted (soft) | Wrong-key decrypt
    Row 2: Error heatmap (per 4×4 block bit errors) + Eve learning curve

  Also prints:
    • PSNR of the Bob reconstruction (target > 30 dB after image-patch training)
    • Per-block error statistics (total, mean, perfect/minor/major)
    • Eve security status (SECURE / PARTIAL BREAK)

  Checkpoint default: ./checkpoints/image_patches_4x4
  (train with:  python main_enhanced.py --mode image_patches
                  --image_dir ./training_images --patch_size 4
                  --msg_size 16 --steps 20000 --seed 13
                  --save_dir ./checkpoints/image_patches_4x4)
=============================================================================
"""

import argparse
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ImageDraw


# ── Synthetic test-image generator ───────────────────────────────────────────

def create_test_image(path: str = "/tmp/test_pattern.png", size: int = 128) -> str:
    """
    Produce a 128×128 grayscale test pattern that exercises both bright and
    dark regions, an edge, and a centre circle.  Useful for verifying that
    soft decoding preserves greyscale (not just hard black/white).
    """
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[:size // 2, :size // 2] = 200   # top-left  : light grey
    arr[:size // 2, size // 2:] = 100   # top-right : mid grey
    arr[size // 2:, :size // 2] =  50   # bot-left  : dark grey
    arr[size // 2:, size // 2:] = 150   # bot-right : mid-light grey

    img  = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    r    = size // 5
    cx, cy = size // 2, size // 2
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    draw.line([(cx, 0),    (cx, size)], fill=0, width=3)
    draw.line([(0,   cy), (size, cy)], fill=0, width=3)
    img.save(path)
    print(f"[Test] Synthetic pattern → {path}  ({size}×{size} px)")
    return path


# ── Main demo ─────────────────────────────────────────────────────────────────

def run_demo(
    image_path:      str,
    checkpoint_dir:  str,
    output_fig:      str,
    eve_steps:       int = 3000,
):
    """
    Full end-to-end demo:
      1. Encrypt with correct key  → ciphertext
      2. Decrypt with correct key  → Bob reconstruction  (soft decoding)
      3. Decrypt with wrong key    → noise (shows key dependency)
      4. Run verify_encryption     → per-block error heatmap + stats
      5. Train fresh Eve           → security status
      6. Save 5-panel figure
    """
    from image_encryptor import ImageEncryptor
    import tensorflow as tf
    from tensorflow import keras
    from main_enhanced import AttackerNet, l1_distance, eve_loss

    print("\n" + "=" * 66)
    print("  ADVERSARIAL NEURAL CRYPTOGRAPHY — Image Encryption Demo")
    print("  Checkpoint : " + checkpoint_dir)
    print("  [RESEARCH DEMONSTRATION — NOT production-grade security]")
    print("=" * 66 + "\n")

    enc = ImageEncryptor(checkpoint_dir=checkpoint_dir)
    key = enc.generate_key()
    print(f"Key (first 8 bits): {key[:8].astype(int).tolist()}\n")

    img_orig = np.array(Image.open(image_path).convert("L"), dtype=np.uint8)
    H, W     = img_orig.shape
    print(f"Image: {image_path}  ({H}×{W} px)\n")

    # ── 1. Encrypt ────────────────────────────────────────────────────────────
    enc_path = "/tmp/_demo_enc.npz"
    t0       = time.time()
    enc.encrypt_image(image_path, key, output_path=enc_path)
    t_enc = time.time() - t0

    # ── 2. Correct-key decrypt (soft decoding) ────────────────────────────────
    dec_path = "/tmp/_demo_dec.png"
    t0       = time.time()
    img_dec  = enc.decrypt_image(enc_path, key, output_path=dec_path)
    t_dec    = time.time() - t0

    mse  = np.mean((img_orig.astype(float) - img_dec.astype(float)) ** 2)
    psnr = float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)

    # ── 3. Wrong-key decrypt ──────────────────────────────────────────────────
    wrong_key = enc.generate_key()
    # Guarantee at least one differing bit
    if np.all(wrong_key == key):
        wrong_key[0] *= -1
    img_wrong = enc.decrypt_image(enc_path, wrong_key, output_path="/tmp/_wrong.png")

    # ── 4. Per-block error stats + heatmap ────────────────────────────────────
    print("\n[Demo] Running verify_encryption …")
    heatmap_path = "/tmp/_demo_heatmap.png"
    stats = enc.verify_encryption(image_path, key, output_heatmap=heatmap_path)
    heatmap_img  = np.array(Image.open(heatmap_path).convert("RGB"))

    # ── 5. Eve security evaluation ────────────────────────────────────────────
    print(f"\n[Demo] Training fresh Eve for {eve_steps} steps against the cipher …")
    data       = np.load(enc_path)
    cipher     = data["ciphertext"]       # (n_blocks, 16)
    msg_size   = enc.msg_size
    gt_blocks, _, _, _ = enc._image_to_blocks(img_orig)
    BATCH      = 512

    fresh_eve  = AttackerNet(msg_size, False, name="demo_eve")
    opt_eve    = keras.optimizers.Adam(0.0008)
    _          = fresh_eve(tf.zeros([1, msg_size]))    # warm-up

    best_err   = float("inf")
    eve_curve  = []

    for step in range(eve_steps):
        idx    = np.random.randint(0, len(cipher), BATCH)
        b_c    = tf.constant(cipher[idx])
        b_orig = tf.constant(gt_blocks[idx])

        with tf.GradientTape() as tape:
            dec  = fresh_eve(b_c)
            loss = eve_loss(b_orig, dec)
        grads = tape.gradient(loss, fresh_eve.trainable_variables)
        opt_eve.apply_gradients(zip(grads, fresh_eve.trainable_variables))

        err = float(tf.reduce_mean(l1_distance(b_orig, dec)).numpy())
        if err < best_err:
            best_err = err

        log_interval = max(eve_steps // 20, 1)
        if step % log_interval == 0:
            eve_curve.append((step, err))
            print(f"  Eve step {step:5d}: {err:.3f} / {msg_size} bits")

    random_base = msg_size / 2.0
    secure      = best_err >= random_base * 0.85
    sec_status  = "✅ SECURE" if secure else "⚠️  PARTIAL BREAK"
    print(f"\n  Eve best error : {best_err:.3f} / {msg_size}  |  {sec_status}")

    # ── 6. Cipher noise visualisation ────────────────────────────────────────
    img_cipher = enc.visualize_ciphertext(enc_path)

    # ── 7. Build figure ───────────────────────────────────────────────────────
    # Layout:
    #   Row 0 (4 panels): Original | Encrypted | Decrypted | Wrong-key
    #   Row 1 (2 panels): Heatmap (wide) | Eve curve
    fig = plt.figure(figsize=(20, 11))
    gs  = gridspec.GridSpec(
        2, 4,
        height_ratios=[1, 1.1],
        hspace=0.38,
        wspace=0.25,
    )

    # ── Row 0: image panels ───────────────────────────────────────────────────
    def show_img(ax, img, title, cmap="gray"):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    show_img(
        fig.add_subplot(gs[0, 0]),
        img_orig,
        f"Original\n({H}×{W} px)",
    )
    show_img(
        fig.add_subplot(gs[0, 1]),
        img_cipher,
        "Encrypted\n(ciphertext → noise)",
    )
    show_img(
        fig.add_subplot(gs[0, 2]),
        img_dec,
        f"Decrypted (soft decode)\n"
        f"PSNR = {psnr:.1f} dB  |  Mean Bob err = {stats['mean_errors_per_block']:.2f} bits",
    )
    show_img(
        fig.add_subplot(gs[0, 3]),
        img_wrong,
        "Wrong-key decrypt\n(should look like noise)",
    )

    # ── Row 1 left: error heatmap (spans 2 columns) ───────────────────────────
    ax_heat = fig.add_subplot(gs[1, :2])
    ax_heat.imshow(heatmap_img)
    ax_heat.set_title(
        f"Per-Block Bit-Error Heatmap  |  "
        f"Perfect: {stats['blocks_perfect']}  "
        f"Minor: {stats['blocks_minor']}  "
        f"Major: {stats['blocks_major']}  "
        f"(Total errors: {stats['total_errors']} / {stats['total_bits']} bits)",
        fontsize=10,
    )
    ax_heat.axis("off")

    # ── Row 1 right: Eve learning curve (spans 2 columns) ────────────────────
    ax_eve = fig.add_subplot(gs[1, 2:])
    xs = [c[0] for c in eve_curve]
    ys = [c[1] for c in eve_curve]
    ax_eve.plot(
        xs, ys,
        color="#E74C3C", linewidth=2,
        label="Eve reconstruction error (real plaintext)",
    )
    ax_eve.axhline(
        random_base,
        color="#95A5A6", linestyle="--", linewidth=1.5,
        label=f"Random-guess baseline ({random_base:.0f} bits)",
    )
    ax_eve.axhline(
        random_base * 0.85,
        color="#E67E22", linestyle=":", linewidth=1.2,
        label=f"Security threshold 85% = {random_base*0.85:.1f} bits",
    )
    ax_eve.set_xlabel("Eve training step", fontsize=11)
    ax_eve.set_ylabel("Reconstruction error (bits)", fontsize=11)
    ax_eve.set_title(
        f"Security Evaluation — Fresh Eve  ({eve_steps} steps)\n"
        f"Best error: {best_err:.2f}/{msg_size} bits  |  {sec_status}",
        fontsize=11,
    )
    ax_eve.legend(fontsize=9)
    ax_eve.grid(True, alpha=0.3)

    # ── Super-title ────────────────────────────────────────────────────────────
    fig.suptitle(
        "Adversarial Neural Cryptography — Image Encryption Demo\n"
        "⚠️  16-bit block cipher — RESEARCH DEMONSTRATION ONLY",
        fontsize=13, color="darkred", fontweight="bold",
    )

    plt.savefig(output_fig, dpi=150, bbox_inches="tight")
    plt.close()

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  RESULTS SUMMARY")
    print("=" * 66)
    print(f"  Image                  : {H}×{W} px")
    print(f"  Encryption time        : {t_enc:.3f}s")
    print(f"  Decryption time        : {t_dec:.3f}s")
    print(f"  PSNR (soft decode)     : {psnr:.1f} dB")
    print(f"  Mean Bob error/block   : {stats['mean_errors_per_block']:.3f} / {msg_size} bits")
    print(f"  Perfect blocks         : {stats['blocks_perfect']} / {stats['total_blocks']}"
          f"  ({100*stats['blocks_perfect']/stats['total_blocks']:.1f}%)")
    print(f"  Eve best error         : {best_err:.3f} / {msg_size} bits"
          f"  (random = {random_base:.0f})")
    print(f"  Security status        : {sec_status}")
    print(f"  Figure saved           : {output_fig}")
    print("=" * 66 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Neural image encryption demo")
    p.add_argument(
        "--image",
        default=None,
        help="Path to input image (any PIL format)",
    )
    p.add_argument(
        "--checkpoint-dir",
        default="./checkpoints/image_patches_4x4",
        dest="checkpoint_dir",
        help="Directory containing alice/bob/eve _final.weights.h5 files "
             "(default: ./checkpoints/image_patches_4x4)",
    )
    p.add_argument(
        "--output",
        default="encryption_demo.png",
        help="Output figure path (default: encryption_demo.png)",
    )
    p.add_argument(
        "--eve-steps",
        type=int,
        default=3000,
        dest="eve_steps",
        help="Steps to train fresh Eve for security evaluation (default: 3000)",
    )
    p.add_argument(
        "--create-test",
        action="store_true",
        dest="create_test",
        help="Generate a 128×128 synthetic test pattern and use it as input",
    )
    args = p.parse_args()

    if args.create_test or args.image is None:
        image_path = create_test_image()
    else:
        image_path = args.image
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            sys.exit(1)

    run_demo(
        image_path     = image_path,
        checkpoint_dir = args.checkpoint_dir,
        output_fig     = args.output,
        eve_steps      = args.eve_steps,
    )
