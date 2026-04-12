"""
=============================================================================
  security_eval.py — Post-Training Eve Robustness Evaluation for Images
=============================================================================
  Trains multiple independent Eve networks from scratch against a frozen
  Alice/Bob cipher applied to a test image. Reports per-run and aggregate
  security metrics consistent with the retrain_eve_robustness() protocol
  in main_enhanced.py.

  Usage:
    python security_eval.py --image test.jpg
    python security_eval.py --image test.jpg --runs 10 --steps 5000
=============================================================================
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image


def run_security_evaluation(
    image_path: str,
    checkpoint_dir: str = "./checkpoints/seed_13",
    n_runs: int = 5,
    steps_per_run: int = 5000,
    output_dir: str = ".",
):
    from image_encryptor import ImageEncryptor
    from main_enhanced import AttackerNet, l1_distance, eve_loss

    print("\n" + "=" * 64)
    print("  SECURITY EVALUATION — Post-Training Eve Robustness")
    print(f"  Image   : {image_path}")
    print(f"  Runs    : {n_runs}  ×  {steps_per_run} steps each")
    print("=" * 64 + "\n")

    # ── Encrypt test image ────────────────────────────────────────────────────
    enc = ImageEncryptor(checkpoint_dir=checkpoint_dir)
    key = enc.generate_key()
    print(f"Key: {key.astype(int).tolist()}\n")

    enc_path = "/tmp/_sec_eval_encrypted.npz"
    enc.encrypt_image(image_path, key, output_path=enc_path)

    # ── Load ciphertext and ground-truth blocks ────────────────────────────────
    data   = np.load(enc_path)
    cipher = data["ciphertext"]           # (n_blocks, 16)
    img_np = np.array(Image.open(image_path).convert("L"))
    blocks, _, _, _ = enc._image_to_blocks(img_np)

    msg_size      = enc.msg_size
    random_base   = msg_size / 2.0
    threshold     = random_base * 0.85
    BATCH         = 512

    run_results   = []   # best Eve error per run
    all_curves    = []   # list of (steps, errors) per run

    for run_idx in range(n_runs):
        print(f"── Run {run_idx + 1}/{n_runs} " + "─" * 44)
        fresh_eve = AttackerNet(msg_size, False, name=f"eve_run{run_idx}")
        opt_eve   = keras.optimizers.Adam(0.0008)
        _ = fresh_eve(tf.zeros([1, msg_size]))   # warm up

        best_err = float("inf")
        curve    = []

        for step in range(steps_per_run):
            idx    = np.random.randint(0, len(cipher), BATCH)
            b_c    = tf.constant(cipher[idx])
            b_orig = tf.constant(blocks[idx])

            with tf.GradientTape() as tape:
                dec  = fresh_eve(b_c)
                loss = eve_loss(b_orig, dec)
            grads = tape.gradient(loss, fresh_eve.trainable_variables)
            opt_eve.apply_gradients(zip(grads, fresh_eve.trainable_variables))

            err = float(tf.reduce_mean(l1_distance(b_orig, dec)).numpy())
            if err < best_err:
                best_err = err
            curve.append(err)

            if step % max(steps_per_run // 10, 1) == 0:
                print(f"  Step {step:5d}/{steps_per_run}  error: {err:.3f} bits")

        status = "✅ SECURE" if best_err >= threshold else "⚠️  PARTIAL BREAK"
        print(f"  Best error: {best_err:.3f} / {msg_size}  {status}\n")
        run_results.append(best_err)
        all_curves.append(curve)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    mean_err    = float(np.mean(run_results))
    secure_runs = sum(1 for e in run_results if e >= threshold)

    print("=" * 64)
    print("  AGGREGATE RESULTS")
    print("=" * 64)
    print(f"  Best errors per run : {[round(e, 3) for e in run_results]}")
    print(f"  Mean best error     : {mean_err:.3f} bits")
    print(f"  Random baseline     : {random_base:.1f} bits")
    print(f"  Security threshold  : {threshold:.1f} bits (85% of baseline)")
    print(f"  Secure runs         : {secure_runs}/{n_runs}")
    overall = "✅ ROBUSTLY SECURE" if secure_runs == n_runs else \
              ("⚠️  PARTIALLY SECURE" if secure_runs > 0 else "❌ INSECURE")
    print(f"  Overall status      : {overall}")
    print("=" * 64)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: bar chart of best errors
    ax = axes[0]
    run_labels = [f"Run {i+1}" for i in range(n_runs)]
    colors     = ["#27AE60" if e >= threshold else "#E74C3C" for e in run_results]
    ax.bar(run_labels, run_results, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(random_base, color="#95A5A6", linestyle="--",
               label=f"Random baseline ({random_base:.0f} bits)")
    ax.axhline(threshold,   color="#E67E22", linestyle=":",
               label=f"Security threshold ({threshold:.1f} bits)")
    ax.set_ylabel("Best Eve error (bits)", fontsize=11)
    ax.set_title(
        f"Per-Run Best Eve Error\n"
        f"{secure_runs}/{n_runs} secure  |  Mean: {mean_err:.2f} bits",
        fontsize=11
    )
    ax.set_ylim(0, msg_size + 1)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    # Right: all learning curves
    ax2 = axes[1]
    xs = list(range(steps_per_run))
    for i, curve in enumerate(all_curves):
        color = "#27AE60" if run_results[i] >= threshold else "#E74C3C"
        ax2.plot(xs, curve, alpha=0.7, linewidth=1.2, color=color,
                 label=f"Run {i+1}: {run_results[i]:.2f}")
    ax2.axhline(random_base, color="#95A5A6", linestyle="--", linewidth=1.4,
                label=f"Random baseline ({random_base:.0f})")
    ax2.axhline(threshold,   color="#E67E22", linestyle=":", linewidth=1.2,
                label=f"Security threshold ({threshold:.1f})")
    ax2.set_xlabel("Training step"); ax2.set_ylabel("Eve reconstruction error (bits)")
    ax2.set_title("Eve Learning Curves — All Runs", fontsize=11)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Security Evaluation: {os.path.basename(image_path)}\n"
        f"[16-bit neural cipher — RESEARCH DEMO]",
        fontsize=12, color="darkred"
    )
    fig_path = os.path.join(output_dir, "security_eval.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot] Security evaluation figure → {fig_path}")

    # ── JSON report ───────────────────────────────────────────────────────────
    import json
    report = {
        "image": image_path,
        "checkpoint_dir": checkpoint_dir,
        "n_runs": n_runs,
        "steps_per_run": steps_per_run,
        "msg_size": msg_size,
        "random_baseline": random_base,
        "security_threshold": threshold,
        "best_errors": run_results,
        "mean_best_error": mean_err,
        "secure_runs": secure_runs,
        "overall_status": overall,
    }
    json_path = os.path.join(output_dir, "security_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Report] JSON report → {json_path}")

    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Neural cipher security evaluation for images")
    p.add_argument("--image",          required=True,      help="Test image path")
    p.add_argument("--checkpoint-dir", default="./checkpoints/seed_13", dest="checkpoint_dir")
    p.add_argument("--runs",           type=int, default=5, help="Number of Eve retraining runs")
    p.add_argument("--steps",          type=int, default=5000, help="Steps per run")
    p.add_argument("--output-dir",     default=".",  dest="output_dir",
                   help="Directory for output plots and report")
    args = p.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    run_security_evaluation(
        image_path     = args.image,
        checkpoint_dir = args.checkpoint_dir,
        n_runs         = args.runs,
        steps_per_run  = args.steps,
        output_dir     = args.output_dir,
    )
