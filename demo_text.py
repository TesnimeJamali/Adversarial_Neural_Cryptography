"""
=============================================================================
  demo_text.py — Interactive Demo for Neural Text Encryption
=============================================================================
  Usage:
    python demo_text.py                          # uses built-in samples
    python demo_text.py --message "Your message" # encrypt a specific message
    python demo_text.py --input message.txt      # encrypt from file
    python demo_text.py --mode random            # use random-mode cipher

  Produces:
    - Console output showing encrypt/decrypt round-trip
    - text_demo.png figure (5-panel layout)
    - Security evaluation with a fresh Eve attacker
=============================================================================
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def run_demo(
    message: str,
    checkpoint_dir: str,
    mode: str,
    eve_steps: int,
    output_fig: str,
):
    from text_encryptor import TextEncryptor
    from main_enhanced import AttackerNet, l1_distance, eve_loss
    from tensorflow import keras
    import tensorflow as tf

    print("\n" + "=" * 66)
    print("  ADVERSARIAL NEURAL CRYPTOGRAPHY — Text Encryption Demo")
    print(f"  Mode       : {mode.upper()}")
    print(f"  Checkpoint : {checkpoint_dir}")
    print("  [RESEARCH DEMONSTRATION — NOT production-grade security]")
    print("=" * 66 + "\n")

    enc = TextEncryptor(checkpoint_dir=checkpoint_dir, mode=mode)
    key = enc.generate_key()
    print(f"Key : {key.astype(int).tolist()}\n")
    print(f"Message ({len(message)} chars):\n  '{message[:80]}{'...' if len(message)>80 else ''}'\n")

    # ── Encrypt ───────────────────────────────────────────────────────────────
    result    = enc.encrypt_text(message, key=key)
    cipher    = result["ciphertext"]

    # ── Correct-key decrypt ───────────────────────────────────────────────────
    decrypted = enc.decrypt_text(cipher, key, original_length=len(message))
    match     = (decrypted == message)

    # ── Wrong-key decrypt ─────────────────────────────────────────────────────
    wrong_key = enc.generate_key()
    if np.all(wrong_key == key):
        wrong_key[0] *= -1
    dec_wrong = enc.decrypt_text(cipher, wrong_key, original_length=len(message))

    print(f"\n  Original  : '{message[:60]}'")
    print(f"  Decrypted : '{decrypted[:60]}'  {'✅ MATCH' if match else '❌ MISMATCH'}")
    print(f"  Wrong key : '{dec_wrong[:60]}'")

    # ── Bit error stats ───────────────────────────────────────────────────────
    blocks_orig = enc._text_to_blocks(message)
    bits_bob    = np.sign(enc.bob(
        tf.concat([
            tf.constant(cipher),
            tf.tile(key[np.newaxis, :], [len(cipher), 1]),
        ], axis=1)
    ).numpy())
    bob_errors  = int(np.sum(bits_bob != blocks_orig))
    total_bits  = blocks_orig.size
    print(f"\n  Bob bit errors : {bob_errors} / {total_bits} ({100*bob_errors/total_bits:.1f}%)")

    # ── Eve security evaluation ───────────────────────────────────────────────
    print(f"\n[Demo] Training fresh Eve for {eve_steps} steps …")
    fresh_eve = AttackerNet(enc.msg_size, False, name="demo_eve")
    opt_eve   = keras.optimizers.Adam(0.0008)
    _ = fresh_eve(tf.zeros([1, enc.msg_size]))

    best_err  = float("inf")
    eve_curve = []
    BATCH     = min(512, len(cipher))

    for step in range(eve_steps):
        idx   = np.random.randint(0, len(cipher), BATCH)
        b_c   = tf.constant(cipher[idx])
        b_msg = tf.constant(blocks_orig[idx])
        with tf.GradientTape() as tape:
            dec  = fresh_eve(b_c)
            loss = eve_loss(b_msg, dec)
        grads = tape.gradient(loss, fresh_eve.trainable_variables)
        opt_eve.apply_gradients(zip(grads, fresh_eve.trainable_variables))
        err = float(tf.reduce_mean(l1_distance(b_msg, dec)).numpy())
        if err < best_err:
            best_err = err
        if step % max(eve_steps // 20, 1) == 0:
            eve_curve.append((step, err))
            print(f"  Eve step {step:5d}: {err:.3f}/{enc.msg_size} bits")

    random_base = enc.msg_size / 2.0
    secure      = best_err >= random_base * 0.85
    sec_status  = "✅ SECURE" if secure else "⚠️  PARTIAL BREAK"

    print(f"\n  Eve best error : {best_err:.3f} / {enc.msg_size}")
    print(f"  Random baseline: {random_base:.1f}")
    print(f"  Status         : {sec_status}")

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.3)

    # ── 1. Original message (top-left) ────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    wrapped = "\n".join(message[i:i+30] for i in range(0, min(len(message), 150), 30))
    ax0.text(0.05, 0.9, "Original Message", fontsize=11, fontweight="bold",
             transform=ax0.transAxes, va="top")
    ax0.text(0.05, 0.75, wrapped, fontsize=9, transform=ax0.transAxes, va="top",
             fontfamily="monospace", color="#2C3E50",
             bbox=dict(boxstyle="round", facecolor="#D6EAF8", alpha=0.6))
    ax0.text(0.05, 0.1, f"{len(message)} chars  |  {len(cipher)} blocks  |  mode={mode}",
             fontsize=8, transform=ax0.transAxes, color="gray")

    # ── 2. Ciphertext (top-centre) ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.axis("off")
    cipher_preview = str(cipher[:3].round(3).tolist())[:120]
    ax1.text(0.05, 0.9, "Ciphertext (first 3 blocks)", fontsize=11, fontweight="bold",
             transform=ax1.transAxes, va="top")
    ax1.text(0.05, 0.75, cipher_preview + " …", fontsize=8, transform=ax1.transAxes,
             va="top", fontfamily="monospace", color="#E74C3C",
             wrap=True,
             bbox=dict(boxstyle="round", facecolor="#FADBD8", alpha=0.5))
    ax1.text(0.05, 0.1, "Continuous real-valued (tanh output)",
             fontsize=8, transform=ax1.transAxes, color="gray")

    # ── 3. Correct-key decryption (top-right) ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    dec_wrapped = "\n".join(decrypted[i:i+30] for i in range(0, min(len(decrypted), 150), 30))
    ax2.text(0.05, 0.9, "Bob's Decryption (correct key)", fontsize=11, fontweight="bold",
             transform=ax2.transAxes, va="top")
    ax2.text(0.05, 0.75, dec_wrapped, fontsize=9, transform=ax2.transAxes, va="top",
             fontfamily="monospace", color="#1E8449",
             bbox=dict(boxstyle="round", facecolor="#D5F5E3", alpha=0.6))
    icon = "✅ Perfect" if bob_errors == 0 else f"⚠️  {bob_errors}/{total_bits} bit errors"
    ax2.text(0.05, 0.1, icon, fontsize=9, transform=ax2.transAxes,
             color="#1E8449" if bob_errors == 0 else "#E67E22")

    # ── 4. Wrong-key decryption (bottom-left) ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis("off")
    wrong_wrapped = "\n".join(dec_wrong[i:i+30] for i in range(0, min(len(dec_wrong), 150), 30))
    ax3.text(0.05, 0.9, "Wrong-Key Decrypt (should be garbage)", fontsize=11,
             fontweight="bold", transform=ax3.transAxes, va="top")
    ax3.text(0.05, 0.75, wrong_wrapped, fontsize=9, transform=ax3.transAxes, va="top",
             fontfamily="monospace", color="#7F8C8D",
             bbox=dict(boxstyle="round", facecolor="#EAECEE", alpha=0.6))
    ax3.text(0.05, 0.1, "Unreadable without correct key ✅",
             fontsize=8, transform=ax3.transAxes, color="gray")

    # ── 5. Bit-error bar (bottom-centre) ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    categories  = ["Bob (correct key)", "Eve (best)", "Random guess"]
    errors      = [bob_errors / total_bits * enc.msg_size,
                   best_err, random_base]
    colors      = ["#27AE60", "#E74C3C", "#95A5A6"]
    bars = ax4.bar(categories, errors, color=colors, edgecolor="white")
    ax4.axhline(random_base, color="#95A5A6", linestyle="--", linewidth=1.2)
    ax4.set_ylabel("Mean error (bits)")
    ax4.set_title("Reconstruction Error Comparison", fontsize=10)
    ax4.set_ylim(0, enc.msg_size + 1)
    for bar, val in zip(bars, errors):
        ax4.text(bar.get_x() + bar.get_width()/2, val + 0.2, f"{val:.2f}",
                 ha="center", va="bottom", fontsize=9)
    ax4.grid(True, alpha=0.3, axis="y")

    # ── 6. Eve learning curve (bottom-right) ──────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    xs = [c[0] for c in eve_curve]; ys = [c[1] for c in eve_curve]
    ax5.plot(xs, ys, color="#E74C3C", linewidth=2, label="Eve reconstruction error")
    ax5.axhline(random_base, color="#95A5A6", linestyle="--", linewidth=1.5,
                label=f"Random baseline ({random_base:.0f} bits)")
    ax5.axhline(random_base * 0.85, color="#E67E22", linestyle=":", linewidth=1.2,
                label=f"Security threshold ({random_base*0.85:.1f} bits)")
    ax5.set_xlabel("Eve training step"); ax5.set_ylabel("Error (bits)")
    ax5.set_title(f"Security — Fresh Eve  ({eve_steps} steps)\n{sec_status}", fontsize=10)
    ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

    fig.suptitle(
        "Adversarial Neural Cryptography — Text Encryption Demo\n"
        "⚠️  16-bit block cipher — RESEARCH DEMONSTRATION ONLY",
        fontsize=13, color="darkred", fontweight="bold",
    )
    plt.savefig(output_fig, dpi=150, bbox_inches="tight")
    plt.close()

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  SUMMARY")
    print("=" * 66)
    print(f"  Message length    : {len(message)} chars")
    print(f"  Blocks            : {len(cipher)} × {enc.msg_size} bits")
    print(f"  Bob bit errors    : {bob_errors} / {total_bits} ({100*bob_errors/total_bits:.2f}%)")
    print(f"  Round-trip match  : {'✅ YES' if match else '❌ NO'}")
    print(f"  Eve best error    : {best_err:.3f} / {enc.msg_size} bits")
    print(f"  Security status   : {sec_status}")
    print(f"  Figure saved      : {output_fig}")
    print("=" * 66 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Neural text encryption interactive demo")
    p.add_argument("--message", default=None, help="Message to encrypt")
    p.add_argument("--input",   default=None, help="Text file to encrypt")
    p.add_argument("--mode",    default="ascii", choices=["ascii", "random"])
    p.add_argument("--checkpoint-dir", default=None, dest="checkpoint_dir")
    p.add_argument("--output",  default="text_demo.png")
    p.add_argument("--eve-steps", type=int, default=3000, dest="eve_steps")
    args = p.parse_args()

    # Determine message
    if args.input:
        with open(args.input) as f:
            message = f.read()
    elif args.message:
        message = args.message
    else:
        message = "The quick brown fox jumps over the lazy dog. Neural cryptography is amazing!"

    # Determine checkpoint
    ckpt_map = {
        "ascii":  "./checkpoints/ascii_quadratic",
        "random": "./checkpoints/seed_13",
    }
    ckpt = args.checkpoint_dir or ckpt_map.get(args.mode, "./checkpoints/seed_13")

    run_demo(
        message        = message,
        checkpoint_dir = ckpt,
        mode           = args.mode,
        eve_steps      = args.eve_steps,
        output_fig     = args.output,
    )
