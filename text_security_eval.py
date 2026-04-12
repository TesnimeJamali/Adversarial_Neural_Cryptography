"""
=============================================================================
  text_security_eval.py — Security Evaluation for Neural Text Cipher
=============================================================================
  Trains N independent Eve networks from scratch against a frozen cipher
  applied to a test message. Reports per-run and aggregate security metrics.

  Usage:
    python text_security_eval.py --message "Secret message"
    python text_security_eval.py --input my_doc.txt --runs 10 --steps 5000
    python text_security_eval.py --mode random --message "Test" --runs 5
=============================================================================
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_text_security_evaluation(
    message: str,
    checkpoint_dir: str,
    mode: str = "ascii",
    n_runs: int = 5,
    steps_per_run: int = 5000,
    output_dir: str = ".",
) -> dict:
    from text_encryptor import TextEncryptor
    from main_enhanced import AttackerNet, l1_distance, eve_loss
    from tensorflow import keras
    import tensorflow as tf

    print("\n" + "=" * 66)
    print("  SECURITY EVALUATION — Post-Training Eve Robustness (Text)")
    print(f"  Mode    : {mode.upper()}")
    print(f"  Message : '{message[:60]}{'...' if len(message)>60 else ''}'")
    print(f"  Runs    : {n_runs}  ×  {steps_per_run} steps each")
    print("=" * 66 + "\n")

    enc = TextEncryptor(checkpoint_dir=checkpoint_dir, mode=mode)
    key = enc.generate_key()
    print(f"Key: {key.astype(int).tolist()}\n")

    result     = enc.encrypt_text(message, key=key)
    cipher     = result["ciphertext"]
    blocks_gt  = enc._text_to_blocks(message)
    msg_size   = enc.msg_size
    random_base = msg_size / 2.0
    threshold   = random_base * 0.85
    BATCH       = min(512, max(1, len(cipher)))

    run_results  = []
    all_curves   = []

    for run_idx in range(n_runs):
        print(f"── Run {run_idx + 1}/{n_runs} " + "─" * 46)
        fresh_eve = AttackerNet(msg_size, False, name=f"eve_run{run_idx}")
        opt_eve   = keras.optimizers.Adam(0.0008)
        _ = fresh_eve(tf.zeros([1, msg_size]))

        best_err = float("inf")
        curve    = []

        for step in range(steps_per_run):
            idx   = np.random.randint(0, len(cipher), BATCH)
            b_c   = tf.constant(cipher[idx])
            b_msg = tf.constant(blocks_gt[idx])
            with tf.GradientTape() as tape:
                dec  = fresh_eve(b_c)
                loss = eve_loss(b_msg, dec)
            grads = tape.gradient(loss, fresh_eve.trainable_variables)
            opt_eve.apply_gradients(zip(grads, fresh_eve.trainable_variables))
            err = float(tf.reduce_mean(l1_distance(b_msg, dec)).numpy())
            if err < best_err:
                best_err = err
            curve.append(err)

            if step % max(steps_per_run // 10, 1) == 0:
                print(f"  Step {step:5d}/{steps_per_run}  error: {err:.3f} bits")

        status = "✅ SECURE" if best_err >= threshold else "⚠️  PARTIAL BREAK"
        print(f"  Best error: {best_err:.3f} / {msg_size}  {status}\n")
        run_results.append(best_err)
        all_curves.append(curve)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    mean_err    = float(np.mean(run_results))
    secure_runs = sum(1 for e in run_results if e >= threshold)
    overall     = ("✅ ROBUSTLY SECURE" if secure_runs == n_runs
                   else ("⚠️  PARTIALLY SECURE" if secure_runs > 0
                         else "❌ INSECURE"))

    print("=" * 66)
    print("  AGGREGATE RESULTS")
    print("=" * 66)
    print(f"  Message length      : {len(message)} chars")
    print(f"  Encoding mode       : {mode}")
    print(f"  Best errors per run : {[round(e, 3) for e in run_results]}")
    print(f"  Mean best error     : {mean_err:.3f} bits")
    print(f"  Random baseline     : {random_base:.1f} bits")
    print(f"  Security threshold  : {threshold:.1f} bits (85% of baseline)")
    print(f"  Secure runs         : {secure_runs}/{n_runs}")
    print(f"  Overall status      : {overall}")
    print("=" * 66)

    # ── Plot ──────────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: bar chart
    ax = axes[0]
    run_labels = [f"Run {i+1}" for i in range(n_runs)]
    colors     = ["#27AE60" if e >= threshold else "#E74C3C" for e in run_results]
    ax.bar(run_labels, run_results, color=colors, edgecolor="white")
    ax.axhline(random_base, color="#95A5A6", linestyle="--",
               label=f"Random baseline ({random_base:.0f} bits)")
    ax.axhline(threshold, color="#E67E22", linestyle=":",
               label=f"Security threshold ({threshold:.1f} bits)")
    ax.set_ylabel("Best Eve error (bits)", fontsize=11)
    ax.set_title(
        f"Per-Run Best Eve Error\n{secure_runs}/{n_runs} secure  |  Mean: {mean_err:.2f} bits",
        fontsize=11
    )
    ax.set_ylim(0, msg_size + 1)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    # Right: all learning curves
    ax2 = axes[1]
    xs  = list(range(steps_per_run))
    for i, curve in enumerate(all_curves):
        color = "#27AE60" if run_results[i] >= threshold else "#E74C3C"
        ax2.plot(xs, curve, alpha=0.7, linewidth=1.2, color=color,
                 label=f"Run {i+1}: {run_results[i]:.2f}")
    ax2.axhline(random_base, color="#95A5A6", linestyle="--", linewidth=1.4,
                label=f"Random baseline ({random_base:.0f})")
    ax2.axhline(threshold, color="#E67E22", linestyle=":", linewidth=1.2,
                label=f"Security threshold ({threshold:.1f})")
    ax2.set_xlabel("Training step"); ax2.set_ylabel("Eve reconstruction error (bits)")
    ax2.set_title("Eve Learning Curves — All Runs", fontsize=11)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    msg_preview = message[:40] + "..." if len(message) > 40 else message
    fig.suptitle(
        f"Text Security Evaluation: '{msg_preview}'\n"
        f"Mode: {mode.upper()}  |  [16-bit neural cipher — RESEARCH DEMO]",
        fontsize=12, color="darkred",
    )
    fig_path = os.path.join(output_dir, "text_security_eval.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot] Security evaluation figure → {fig_path}")

    # ── JSON report ───────────────────────────────────────────────────────────
    import json
    report = {
        "message_preview": message[:80],
        "message_length":  len(message),
        "mode":            mode,
        "checkpoint_dir":  checkpoint_dir,
        "n_runs":          n_runs,
        "steps_per_run":   steps_per_run,
        "msg_size":        msg_size,
        "random_baseline": random_base,
        "security_threshold": threshold,
        "best_errors":     run_results,
        "mean_best_error": mean_err,
        "secure_runs":     secure_runs,
        "overall_status":  overall,
    }
    json_path = os.path.join(output_dir, "text_security_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Report] JSON report → {json_path}")

    return report


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    p = argparse.ArgumentParser(description="Neural text cipher security evaluation")
    p.add_argument("--message",        default=None, help="Message to evaluate")
    p.add_argument("--input",          default=None, help="Text file to evaluate")
    p.add_argument("--mode",           default="ascii", choices=["ascii", "random"])
    p.add_argument("--checkpoint-dir", default=None, dest="checkpoint_dir")
    p.add_argument("--runs",           type=int, default=5)
    p.add_argument("--steps",          type=int, default=5000)
    p.add_argument("--output-dir",     default=".", dest="output_dir")
    args = p.parse_args()

    if args.input:
        with open(args.input) as f:
            message = f.read()
    elif args.message:
        message = args.message
    else:
        message = "The quick brown fox jumps over the lazy dog."

    ckpt_map = {
        "ascii":  "./checkpoints/ascii_quadratic",
        "random": "./checkpoints/seed_13",
    }
    ckpt = args.checkpoint_dir or ckpt_map.get(args.mode, "./checkpoints/seed_13")

    run_text_security_evaluation(
        message        = message,
        checkpoint_dir = ckpt,
        mode           = args.mode,
        n_runs         = args.runs,
        steps_per_run  = args.steps,
        output_dir     = args.output_dir,
    )
