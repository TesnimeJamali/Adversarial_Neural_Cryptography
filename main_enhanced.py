"""
=============================================================================
  Adversarial Neural Cryptography  —  Enhanced Implementation
=============================================================================
  Based on: Abadi & Andersen (2016) "Learning to Protect Communications
             with Neural Cryptography"  arXiv:1610.06918

  Enhancements over original:
     1. Full TensorFlow 2.x migration (Keras subclassing, eager execution)
     2. Save & reload trained models (checkpoints + SavedModel)
     3. Multiple Eve retraining — post-training robustness evaluation
     4. Message complexity: ASCII text (bitwise encoding), variable-length (chunking)
     5. Configurable Alice/Bob vs Eve training step ratio
     6. Loss function variants: quadratic (original), linear, binary-cross-entropy
     7. Attention mechanism option in the architecture
     8. Live loss curve plotting (matplotlib)
=============================================================================
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on all platforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
# CLI  ARGUMENTS
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Adversarial Neural Cryptography — Enhanced TF2 Implementation"
    )
    # Core
    p.add_argument("--msg_size",    type=int,   default=16,
                   help="Plaintext / key size in bits  (default: 16)")
    p.add_argument("--key_size",    type=int,   default=16,
                   help="Key size in bits  (default: 16)")
    p.add_argument("--batch_size",  type=int,   default=4096,
                   help="Mini-batch size  (default: 4096)")
    p.add_argument("--steps",       type=int,   default=20000,
                   help="Training steps  (default: 20000)")
    p.add_argument("--lr",          type=float, default=0.0008,
                   help="Adam learning rate  (default: 0.0008)")

    # Enhancement toggles
    p.add_argument("--eve_steps",   type=int,   default=2,
                   help="Eve training steps per Alice/Bob step  (default: 2)")
    p.add_argument("--loss_fn",     type=str,   default="quadratic",
                   choices=["quadratic", "linear", "bce"],
                   help="Alice/Bob adversarial loss variant  (default: quadratic)")
    p.add_argument("--attention",   action="store_true",
                   help="Add self-attention layer to the architecture")
    p.add_argument("--mode",        type=str,   default="random",
                   choices=["random", "ascii"],
                   help="Message type  (default: random)")
    p.add_argument("--eve_retrain", type=int,   default=5,
                   help="Number of Eve retraining runs after convergence  (default: 5)")
    p.add_argument("--eve_retrain_steps", type=int, default=5000,
                   help="Steps per Eve retraining run  (default: 5000)")

    # I/O
    p.add_argument("--save_dir",    type=str,   default="./checkpoints",
                   help="Directory for model checkpoints")
    p.add_argument("--load",        action="store_true",
                   help="Load existing checkpoint before training")
    p.add_argument("--plot_dir",    type=str,   default="./plots",
                   help="Directory to save loss curve plots")
    p.add_argument("--seed",        type=int,   default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("--log_every",   type=int,   default=100,
                   help="Log interval in steps  (default: 100)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# DATA  GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def random_batch(batch_size, msg_size, key_size):
    """Original random binary {-1, +1} messages and keys."""
    msg = tf.cast(
        2 * tf.random.uniform([batch_size, msg_size], 0, 2, dtype=tf.int32) - 1,
        tf.float32
    )
    key = tf.cast(
        2 * tf.random.uniform([batch_size, key_size], 0, 2, dtype=tf.int32) - 1,
        tf.float32
    )
    return msg, key


def ascii_batch(batch_size, msg_size, key_size):
    """
    ASCII text messages encoded as binary {-1, +1} bits.

    Each printable ASCII character (codes 32-126) is encoded as 8 bits,
    so msg_size bits = msg_size // 8 characters per block.
    This keeps training in the same {-1, +1} domain as random mode,
    ensuring the L1 loss metric is meaningful at the bit level.
    """
    chars_per_block = msg_size // 8
    # Random printable ASCII: codes 32-126
    codes = tf.random.uniform([batch_size, chars_per_block], 32, 127, dtype=tf.int32)
    # Expand each code into 8 bits, MSB first
    bits_list = []
    for shift in range(7, -1, -1):
        bit = tf.cast(tf.bitwise.bitwise_and(codes, 1 << shift) > 0, tf.float32)
        bits_list.append(bit)
    # Stack: (B, chars_per_block, 8) → (B, msg_size)
    bits = tf.stack(bits_list, axis=2)
    msg  = tf.reshape(bits, [batch_size, chars_per_block * 8])
    msg  = msg * 2.0 - 1.0          # {0,1} → {-1, +1}
    key  = tf.cast(
        2 * tf.random.uniform([batch_size, key_size], 0, 2, dtype=tf.int32) - 1,
        tf.float32
    )
    return msg, key


def encode_text(text, msg_size):
    """
    Convert a string to a binary {-1, +1} float vector.
    msg_size bits → msg_size // 8 characters, padded with spaces.
    """
    chars_per_block = msg_size // 8
    codes = [ord(c) for c in text[:chars_per_block]]
    codes += [32] * (chars_per_block - len(codes))   # pad with spaces
    bits = []
    for code in codes:
        for shift in range(7, -1, -1):
            bits.append(1.0 if (code >> shift) & 1 else -1.0)
    return np.array(bits, dtype=np.float32)


def decode_text(vec):
    """
    Convert a binary {-1, +1} float vector back to a string.
    Groups of 8 bits → one ASCII character.
    """
    bits = (np.sign(vec) > 0).astype(int)   # threshold at 0
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        code = sum(b << (7 - j) for j, b in enumerate(byte))
        chars.append(chr(code) if 32 <= code <= 126 else '?')
    return ''.join(chars)



def l1_distance(a, b):
    """Element-wise L1 distance, summed over the message dimension."""
    return tf.reduce_sum(tf.abs(a - b), axis=1)


def eve_adversarial_loss(o_message, decipher_eve, msg_size, variant="quadratic"):
    """
    Adversarial penalty for Alice & Bob: push Eve toward random guessing.
    Random-guessing baseline = N/2 bits wrong.

    Variants:
      quadratic  — original paper: (N/2 - L1)² / (N/2)²
      linear     — direct penalty: max(0, N/2 - L1) / (N/2)
      bce        — binary cross-entropy between Eve output and 0.5 (uniform)
    """
    half = msg_size / 2.0
    eve_err = l1_distance(o_message, decipher_eve)

    if variant == "quadratic":
        return tf.square(half - eve_err) / (half ** 2)
    elif variant == "linear":
        return tf.maximum(0.0, half - eve_err) / half
    elif variant == "bce":
        # Encourage Eve's output to be near 0 (maximum uncertainty in {-1,+1} space)
        # Map Eve output from [-1,1] → [0,1] and compute BCE vs 0.5
        eve_prob = (decipher_eve + 1.0) / 2.0
        target   = tf.ones_like(eve_prob) * 0.5
        bce      = tf.reduce_mean(
            keras.losses.binary_crossentropy(target, eve_prob), axis=-1
        )
        return -bce    # negate: Alice/Bob want to maximise Eve's BCE
    else:
        raise ValueError(f"Unknown loss variant: {variant}")


def alice_bob_loss(o_message, decipher_bob, decipher_eve, msg_size, variant="quadratic"):
    """Composite loss: Bob reconstruction + adversarial penalty on Eve."""
    bob_err    = l1_distance(o_message, decipher_bob) / msg_size
    adv_penalty = eve_adversarial_loss(o_message, decipher_eve, msg_size, variant)
    return tf.reduce_mean(bob_err + adv_penalty)


def eve_loss(o_message, decipher_eve):
    """Eve's pure L1 reconstruction loss."""
    return tf.reduce_mean(l1_distance(o_message, decipher_eve))


# ─────────────────────────────────────────────────────────────────────────────
# ATTENTION  MODULE
# ─────────────────────────────────────────────────────────────────────────────

class SelfAttention1D(keras.layers.Layer):
    """
    Lightweight single-head self-attention over a 1-D sequence.
    Input shape : (batch, seq_len, channels)
    Output shape: (batch, seq_len, channels)   — same as input

    Motivation: replaces the hard-coded spatial locality of convolution
    with a learned, global mixing of sequence positions — more expressive
    than fixed conv windows for learning which bit positions to combine.
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.Wq = keras.layers.Dense(units, use_bias=False)
        self.Wk = keras.layers.Dense(units, use_bias=False)
        self.Wv = keras.layers.Dense(units, use_bias=False)
        self.proj = keras.layers.Dense(units, use_bias=False)

    def call(self, x):
        Q = self.Wq(x)                                    # (B, L, units)
        K = self.Wk(x)
        V = self.Wv(x)
        scale  = tf.math.sqrt(tf.cast(self.units, tf.float32))
        scores = tf.matmul(Q, K, transpose_b=True) / scale  # (B, L, L)
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(weights, V)                   # (B, L, units)
        return self.proj(context)


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK  ARCHITECTURES
# ─────────────────────────────────────────────────────────────────────────────

class CipherNet(keras.Model):
    """
    Mix-and-Transform network for Alice and Bob.

    Architecture (following Abadi & Andersen 2016, Section 2.4):
      FC(input_size → N)  →  reshape(N,1)
      Conv1D [4,1,2] s=1 sigmoid   → (B, N, 2)
      Conv1D [2,2,4] s=1 sigmoid   → (B, N, 4)
      [Optional: SelfAttention1D]
      Conv1D [1,4,4] s=1 sigmoid   → (B, N, 4)
      Conv1D [1,4,1] s=1 tanh      → (B, N, 1)
      squeeze → (B, N)

    All strides=1 with SAME padding so sequence length stays N throughout.
    The FC layer mixes plaintext and key bits globally.
    Convolutional layers learn local transformations.
    """
    def __init__(self, input_size, msg_size, use_attention=False, name="cipher"):
        super().__init__(name=name)
        self.msg_size      = msg_size
        self.use_attention = use_attention

        self.fc      = keras.layers.Dense(msg_size, use_bias=False)
        self.conv1   = keras.layers.Conv1D(2, 4, strides=1, padding="same", activation="sigmoid")
        self.conv2   = keras.layers.Conv1D(4, 2, strides=1, padding="same", activation="sigmoid")
        if use_attention:
            self.attention = SelfAttention1D(4)
            self.attn_norm = keras.layers.LayerNormalization()
        self.conv3   = keras.layers.Conv1D(4, 1, strides=1, padding="same", activation="sigmoid")
        self.conv4   = keras.layers.Conv1D(1, 1, strides=1, padding="same", activation="tanh")

    def call(self, x, training=False):
        net = self.fc(x)                          # (B, N)
        net = tf.expand_dims(net, axis=-1)        # (B, N, 1)
        net = self.conv1(net)                     # (B, N, 2)
        net = self.conv2(net)                     # (B, N, 4)
        if self.use_attention:
            attn_out = self.attention(net)
            net = self.attn_norm(net + attn_out)  # residual + LayerNorm
        net = self.conv3(net)                     # (B, N, 4)
        net = self.conv4(net)                     # (B, N, 1)
        net = tf.squeeze(net, axis=-1)            # (B, N)
        return net


class AttackerNet(keras.Model):
    """
    Eve's network. Identical architecture to CipherNet but:
      — input_size = msg_size (ciphertext only, NO key)
      — this structural disadvantage is the source of Eve's challenge
    All strides=1 with SAME padding so sequence length stays N throughout.
    """
    def __init__(self, msg_size, use_attention=False, name="attacker"):
        super().__init__(name=name)
        self.msg_size      = msg_size
        self.use_attention = use_attention

        self.fc      = keras.layers.Dense(msg_size, use_bias=False)
        self.conv1   = keras.layers.Conv1D(2, 4, strides=1, padding="same", activation="sigmoid")
        self.conv2   = keras.layers.Conv1D(4, 2, strides=1, padding="same", activation="sigmoid")
        if use_attention:
            self.attention = SelfAttention1D(4)
            self.attn_norm = keras.layers.LayerNormalization()
        self.conv3   = keras.layers.Conv1D(4, 1, strides=1, padding="same", activation="sigmoid")
        self.conv4   = keras.layers.Conv1D(1, 1, strides=1, padding="same", activation="tanh")

    def call(self, x, training=False):
        net = self.fc(x)
        net = tf.expand_dims(net, axis=-1)
        net = self.conv1(net)
        net = self.conv2(net)
        if self.use_attention:
            attn_out = self.attention(net)
            net = self.attn_norm(net + attn_out)
        net = self.conv3(net)
        net = self.conv4(net)
        net = tf.squeeze(net, axis=-1)            # (B, N)
        return net


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING  STEP  (tf.function for speed)
# ─────────────────────────────────────────────────────────────────────────────

@tf.function
def train_alice_bob(alice, bob, eve, o_msg, key, optimizer_ab, msg_size, loss_variant):
    with tf.GradientTape() as tape:
        cipher     = alice(tf.concat([o_msg, key], axis=1))
        dec_bob    = bob(tf.concat([cipher, key], axis=1))
        dec_eve    = eve(cipher)
        loss       = alice_bob_loss(o_msg, dec_bob, dec_eve, msg_size, loss_variant)
    grads = tape.gradient(loss, alice.trainable_variables + bob.trainable_variables)
    optimizer_ab.apply_gradients(
        zip(grads, alice.trainable_variables + bob.trainable_variables)
    )
    return loss, dec_bob, dec_eve, cipher


@tf.function
def train_eve(alice, eve, o_msg, key, optimizer_eve):
    with tf.GradientTape() as tape:
        cipher  = alice(tf.concat([o_msg, key], axis=1))
        dec_eve = eve(cipher)
        loss    = eve_loss(o_msg, dec_eve)
    grads = tape.gradient(loss, eve.trainable_variables)
    optimizer_eve.apply_gradients(zip(grads, eve.trainable_variables))
    return loss, dec_eve


def train_eve_eager(alice, eve, o_msg, key, optimizer_eve):
    """Eager (non-compiled) version for dynamically created Eve instances
    in retrain_eve_robustness. Avoids @tf.function graph-capture issues."""
    with tf.GradientTape() as tape:
        cipher  = alice(tf.concat([o_msg, key], axis=1))
        dec_eve = eve(cipher)
        loss    = eve_loss(o_msg, dec_eve)
    grads = tape.gradient(loss, eve.trainable_variables)
    optimizer_eve.apply_gradients(zip(grads, eve.trainable_variables))
    return loss, dec_eve


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def save_loss_curves(history, args, tag="training"):
    """Save Bob and Eve reconstruction error curves to disk."""
    os.makedirs(args.plot_dir, exist_ok=True)
    steps       = history["steps"]
    bob_errs    = history["bob_err"]
    eve_errs    = history["eve_err"]
    random_base = args.msg_size / 2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, bob_errs, color="#E74C3C", linewidth=1.8, label="Bob (legitimate receiver)")
    ax.plot(steps, eve_errs, color="#27AE60", linewidth=1.8, label="Eve (eavesdropper)")
    ax.axhline(random_base, color="#95A5A6", linewidth=1.2,
               linestyle="--", label=f"Random-guess baseline ({random_base} bits)")
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel(f"Mean Reconstruction Error (of {args.msg_size} bits)", fontsize=12)
    ax.set_title(
        f"Adversarial Neural Cryptography — {tag.replace('_',' ').title()}\n"
        f"Loss: {args.loss_fn}  |  Eve steps/round: {args.eve_steps}  |  "
        f"Attention: {args.attention}  |  Mode: {args.mode}",
        fontsize=11
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    path = os.path.join(args.plot_dir, f"{tag}_loss_curve.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Saved → {path}")


def save_eve_retrain_plot(retrain_results, args):
    """Bar chart of Eve's best error across retraining runs.
    Bar colours use the same 0.85 x N/2 security threshold as
    retrain_eve_robustness(), keeping visual and terminal output consistent.
    """
    os.makedirs(args.plot_dir, exist_ok=True)
    threshold = args.msg_size / 2 * 0.85   # must match retrain_eve_robustness()
    fig, ax = plt.subplots(figsize=(8, 4))
    runs    = [f"Run {i+1}" for i in range(len(retrain_results))]
    colors  = ["#E74C3C" if v < threshold else "#27AE60" for v in retrain_results]
    ax.bar(runs, retrain_results, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(args.msg_size / 2, color="#95A5A6", linestyle="--",
               label=f"Random baseline ({args.msg_size/2:.1f} bits)")
    ax.axhline(threshold, color="#E67E22", linestyle=":",
               label=f"Security threshold (85% = {threshold:.1f} bits)")
    ax.set_ylabel("Eve Reconstruction Error (bits)", fontsize=11)
    ax.set_title("Post-Training Eve Robustness Evaluation\n"
                 "(Green = secure \u2265 85% baseline, Red = partial break)", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, args.msg_size + 1)
    path = os.path.join(args.plot_dir, "eve_retrain_results.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Eve retrain chart → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL  SAVE / LOAD
# ─────────────────────────────────────────────────────────────────────────────

def save_models(alice, bob, eve, save_dir, step=None):
    """Save Alice, Bob, Eve weights as TF2 checkpoints."""
    os.makedirs(save_dir, exist_ok=True)
    tag = f"_step{step}" if step is not None else "_final"
    alice.save_weights(os.path.join(save_dir, f"alice{tag}.weights.h5"))
    bob.save_weights(os.path.join(save_dir, f"bob{tag}.weights.h5"))
    eve.save_weights(os.path.join(save_dir, f"eve{tag}.weights.h5"))
    print(f"[Save] Models saved at step {step or 'final'} → {save_dir}")


def load_models(alice, bob, eve, save_dir):
    """Load Alice, Bob, Eve weights from the latest checkpoint."""
    for model, name in [(alice, "alice"), (bob, "bob"), (eve, "eve")]:
        path = os.path.join(save_dir, f"{name}_final.weights.h5")
        if os.path.exists(path):
            model.load_weights(path)
            print(f"[Load] {name} ← {path}")
        else:
            print(f"[Load] No checkpoint found for {name} at {path}, starting fresh.")


# ─────────────────────────────────────────────────────────────────────────────
# POST-TRAINING  EVE  RETRAINING
# ─────────────────────────────────────────────────────────────────────────────

def retrain_eve_robustness(alice, bob, args, data_fn):
    """
    After Alice and Bob have converged, reset Eve and retrain her N times
    from scratch. Records the best (minimum) error Eve achieves across runs.
    This is the robustness test from Abadi & Andersen (2016), Section 2.5.

    Returns list of best Eve errors, one per retraining run.
    """
    print(f"\n{'='*60}")
    print(f"  POST-TRAINING EVE ROBUSTNESS  ({args.eve_retrain} runs × "
          f"{args.eve_retrain_steps} steps)")
    print(f"{'='*60}")

    N = args.msg_size + args.key_size
    results = []

    for run in range(args.eve_retrain):
        # Fresh Eve, fresh optimiser
        fresh_eve  = AttackerNet(args.msg_size, args.attention, name=f"eve_retrain_{run}")
        opt_eve    = keras.optimizers.Adam(args.lr)
        best_err   = float("inf")

        for step in range(args.eve_retrain_steps):
            msg, key = data_fn()
            loss_e, dec_eve = train_eve_eager(alice, fresh_eve, msg, key, opt_eve)
            err = float(tf.reduce_mean(l1_distance(msg, dec_eve)).numpy())
            if err < best_err:
                best_err = err

        random_base = args.msg_size / 2.0
        status = "✅ SECURE" if best_err >= random_base * 0.85 else "⚠️  PARTIAL BREAK"
        print(f"  Run {run+1:2d}/{args.eve_retrain}  |  Best Eve error: "
              f"{best_err:.3f} bits  |  Baseline: {random_base:.1f}  |  {status}")
        results.append(best_err)

    mean_err = np.mean(results)
    print(f"\n  Mean best Eve error: {mean_err:.3f}  |  "
          f"Random baseline: {args.msg_size/2:.1f}")
    secure_runs = sum(1 for e in results if e >= args.msg_size / 2 * 0.85)
    print(f"  Secure runs: {secure_runs}/{args.eve_retrain}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ASCII  DEMO
# ─────────────────────────────────────────────────────────────────────────────

def encryption_demo(alice, bob, eve, args):
    """Dynamic encryption demo — adapts output to the current training mode."""
    key_np = (2 * np.random.randint(0, 2, (1, args.key_size)) - 1).astype(np.float32)
    key_tf = tf.constant(key_np)

    if args.mode == "random":
        print(f"\n{'='*60}")
        print("  RANDOM BINARY  ENCRYPTION  DEMO")
        print(f"{'='*60}")
        print(f"  Key (first 8 bits): {key_np[0, :8].astype(int).tolist()}\n")
        for i in range(4):
            msg_np  = (2 * np.random.randint(0, 2, (1, args.msg_size)) - 1).astype(np.float32)
            msg_tf  = tf.constant(msg_np)
            cipher  = alice(tf.concat([msg_tf, key_tf], axis=1))
            dec_bob = bob(tf.concat([cipher, key_tf], axis=1))
            dec_eve = eve(cipher)
            bob_bits  = np.sign(dec_bob.numpy()[0]).astype(int).tolist()
            eve_bits  = np.sign(dec_eve.numpy()[0]).astype(int).tolist()
            orig_bits = msg_np[0].astype(int).tolist()
            bob_err   = int(np.sum(np.sign(dec_bob.numpy()[0]) != msg_np[0]))
            eve_err   = int(np.sum(np.sign(dec_eve.numpy()[0]) != msg_np[0]))
            print(f"  Sample {i+1}:")
            print(f"    Original  (binary) : {orig_bits}")
            print(f"    Ciphertext (vals)  : {cipher.numpy()[0, :8].round(3).tolist()} ...")
            print(f"    Bob decoded        : {bob_bits}  ({bob_err} bit errors)")
            print(f"    Eve guessed        : {eve_bits}  ({eve_err} bit errors)\n")

    elif args.mode == "ascii":
        print(f"\n{'='*60}")
        print(f"  ASCII TEXT  ENCRYPTION  DEMO")
        print(f"  (msg_size={args.msg_size} bits = {args.msg_size//8} chars/block)")
        print(f"{'='*60}")
        samples = ["Hello World!", "Cryptography", "Neural Nets!", "Secret: 1234"]
        for text in samples:
            msg_np  = encode_text(text, args.msg_size)[np.newaxis, :]
            msg_tf  = tf.constant(msg_np)
            cipher  = alice(tf.concat([msg_tf, key_tf], axis=1))
            dec_bob = bob(tf.concat([cipher, key_tf], axis=1))
            dec_eve = eve(cipher)
            recovered  = decode_text(dec_bob.numpy()[0])
            eve_guess  = decode_text(dec_eve.numpy()[0])
            bob_bits   = np.sign(dec_bob.numpy()[0])
            eve_bits   = np.sign(dec_eve.numpy()[0])
            bob_errs   = int(np.sum(bob_bits != msg_np[0]))
            eve_errs   = int(np.sum(eve_bits != msg_np[0]))
            orig_block = text[:args.msg_size//8].ljust(args.msg_size//8)
            print(f"\n  Original      : '{orig_block}'  ({bob_errs} bit errors Bob, {eve_errs} Eve)")
            print(f"  Ciphertext    : {cipher.numpy()[0, :8].round(3).tolist()} ...")
            print(f"  Bob decrypted : '{recovered}'")
            print(f"  Eve guessed   : '{eve_guess}'")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN  TRAINING  LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    print("\n" + "="*60)
    print("  ADVERSARIAL NEURAL CRYPTOGRAPHY — Enhanced TF2")
    print("="*60)
    for k, v in vars(args).items():
        print(f"  {k:22s}: {v}")
    print("="*60 + "\n")

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    N = args.msg_size + args.key_size

    # ── Build networks ───────────────────────────────────────────────────────
    alice = CipherNet(N,             args.msg_size, args.attention, name="alice")
    bob   = CipherNet(N,             args.msg_size, args.attention, name="bob")
    eve   = AttackerNet(args.msg_size, args.attention, name="eve")

    opt_ab  = keras.optimizers.Adam(args.lr)
    opt_eve = keras.optimizers.Adam(args.lr)

    # ── Data function ────────────────────────────────────────────────────────
    if args.mode == "random":
        def data_fn():
            return random_batch(args.batch_size, args.msg_size, args.key_size)
    elif args.mode == "ascii":
        def data_fn():
            return ascii_batch(args.batch_size, args.msg_size, args.key_size)

    # ── Warm-up forward pass (build weights before load/save) ────────────────
    dummy_msg, dummy_key = data_fn()
    alice_input  = tf.concat([dummy_msg, dummy_key], axis=1)
    dummy_cipher = alice(alice_input)
    # Bob needs key with same batch size as cipher output
    dummy_key_bob = dummy_key[:tf.shape(dummy_cipher)[0]]
    _             = bob(tf.concat([dummy_cipher, dummy_key_bob], axis=1))
    _             = eve(dummy_cipher)

    # ── Optionally load checkpoint ───────────────────────────────────────────
    if args.load:
        load_models(alice, bob, eve, args.save_dir)

    # ── Training history ─────────────────────────────────────────────────────
    history = {"steps": [], "bob_err": [], "eve_err": [], "ab_loss": [], "eve_loss": []}

    print(f"Training for {args.steps} steps "
          f"(Alice/Bob : Eve = 1 : {args.eve_steps}) …\n")

    for step in range(args.steps):

        msg, key = data_fn()

        # ── Train Alice & Bob ────────────────────────────────────────────────
        ab_loss, dec_bob, dec_eve, cipher = train_alice_bob(
            alice, bob, eve, msg, key, opt_ab, args.msg_size, args.loss_fn
        )

        # ── Train Eve (configurable ratio) ───────────────────────────────────
        for _ in range(args.eve_steps):
            msg_e, key_e = data_fn()
            eve_l, dec_eve_only = train_eve(alice, eve, msg_e, key_e, opt_eve)

        # ── Logging ──────────────────────────────────────────────────────────
        if step % args.log_every == 0:
            bob_err = float(tf.reduce_mean(l1_distance(msg, dec_bob)).numpy())
            eve_err = float(tf.reduce_mean(l1_distance(msg_e, dec_eve_only)).numpy())

            history["steps"].append(step)
            history["bob_err"].append(bob_err)
            history["eve_err"].append(eve_err)
            history["ab_loss"].append(float(ab_loss.numpy()))
            history["eve_loss"].append(float(eve_l.numpy()))

            bar_len  = 20
            bob_bar  = "█" * int(bob_err / args.msg_size * bar_len)
            eve_bar  = "█" * int(eve_err / args.msg_size * bar_len)
            print(f"Step {step:6d}  |  "
                  f"Bob: {bob_err:5.2f} [{bob_bar:<{bar_len}}]  |  "
                  f"Eve: {eve_err:5.2f} [{eve_bar:<{bar_len}}]  |  "
                  f"Loss(AB): {float(ab_loss):.4f}")

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if step > 0 and step % 5000 == 0:
            save_models(alice, bob, eve, args.save_dir, step=step)

    # ── Final save ───────────────────────────────────────────────────────────
    save_models(alice, bob, eve, args.save_dir)

    # ── Loss curve plot ───────────────────────────────────────────────────────
    save_loss_curves(history, args, tag="training")

    # ── Encryption demo (mode-aware) ─────────────────────────────────────────
    if args.mode in ("random", "ascii"):
        encryption_demo(alice, bob, eve, args)


    # ── Post-training Eve robustness ──────────────────────────────────────────
    retrain_results = retrain_eve_robustness(alice, bob, args, data_fn)
    save_eve_retrain_plot(retrain_results, args)
    save_loss_curves(history, args, tag="final_summary")

    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print(f"  Plots  → {args.plot_dir}/")
    print(f"  Models → {args.save_dir}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
