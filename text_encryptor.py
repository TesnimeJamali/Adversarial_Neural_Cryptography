"""
=============================================================================
  text_encryptor.py — Neural Text Encryption Using Trained Cipher Models
=============================================================================
  Encrypts and decrypts arbitrary text using trained Alice/Bob models from
  main_enhanced.py (Abadi & Andersen 2016).

  Two encoding modes:
    ascii  — 8 bits per character (MSB first), 2 chars per 16-bit block
    random — 16 bits per character (less efficient, works for any text)

  NOTE: RESEARCH DEMONSTRATION — NOT production-grade security.
        16-bit keys are trivially brute-forceable. ASCII mode has MSB bias
        that slightly weakens post-training robustness.
=============================================================================
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import tensorflow as tf

# ── Import models from main_enhanced.py ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_enhanced import CipherNet, AttackerNet, encode_text, decode_text


class TextEncryptor:
    """
    Encrypt and decrypt arbitrary text using trained neural cipher models.

    Block size  : 16 bits.
    ASCII mode  : 8 bits/char → 2 chars per block.
    Random mode : 16 bits/char → 1 char per block.
    Key         : 16-bit {-1, +1} vector, same for every block.
    """

    SUPPORTED_MODES = ("ascii", "random")

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints/seed_13",
        mode: str = "random",
        msg_size: int = 16,
        key_size: int = 16,
    ):
        """
        Load Alice and Bob models from checkpoint directory.

        Args:
            checkpoint_dir : Path containing alice/bob _final.weights.h5
            mode           : "ascii" or "random" encoding
            msg_size       : Message block size in bits (default 16)
            key_size       : Key size in bits (default 16)
        """
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"mode must be one of {self.SUPPORTED_MODES}, got '{mode}'")

        self.checkpoint_dir = checkpoint_dir
        self.mode = mode
        self.msg_size = msg_size
        self.key_size = key_size

        # Encoding width: bits per character
        self.bits_per_char = 8 if mode == "ascii" else 16
        # How many characters fit in one 16-bit block
        self.chars_per_block = msg_size // self.bits_per_char

        self._build_and_load_models()

    # ── Model construction ────────────────────────────────────────────────────

    def _build_and_load_models(self):
        N = self.msg_size + self.key_size
        self.alice = CipherNet(N, self.msg_size, use_attention=False, name="alice")
        self.bob   = CipherNet(N, self.msg_size, use_attention=False, name="bob")

        # Warm-up forward pass — required before load_weights for subclassed models
        dummy_msg = tf.zeros([1, self.msg_size])
        dummy_key = tf.zeros([1, self.key_size])
        dummy_c   = self.alice(tf.concat([dummy_msg, dummy_key], axis=1))
        _         = self.bob(tf.concat([dummy_c, dummy_key], axis=1))

        for model, name in [(self.alice, "alice"), (self.bob, "bob")]:
            path = os.path.join(self.checkpoint_dir, f"{name}_final.weights.h5")
            if os.path.exists(path):
                model.load_weights(path)
                print(f"[Load] {name} ← {path}")
            else:
                warnings.warn(
                    f"Checkpoint not found: {path}\n"
                    f"Using random weights — results will be garbage.",
                    UserWarning,
                )

    # ── Key utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def generate_key(key_size: int = 16) -> np.ndarray:
        """Generate a random {-1, +1} key of the given size."""
        return np.random.choice([-1.0, 1.0], size=key_size).astype(np.float32)

    @staticmethod
    def save_key(key: np.ndarray, path: str):
        """Save key as comma-separated {-1,+1} values."""
        line = ",".join(str(int(x)) for x in key)
        with open(path, "w") as f:
            f.write(line + "\n")
        print(f"[Key] Saved → {path}")

    @staticmethod
    def load_key(path: str) -> np.ndarray:
        """Load key from comma-separated text file."""
        with open(path) as f:
            line = f.read().strip()
        arr = np.array([float(x.strip()) for x in line.split(",")], dtype=np.float32)
        if not np.all(np.isin(arr, [-1.0, 1.0])):
            raise ValueError("Key file must contain only +1 and -1 values.")
        return arr

    # ── Text ↔ bits conversion ────────────────────────────────────────────────

    def text_to_bits(self, text: str) -> np.ndarray:
        """
        Convert text to a flat array of {-1, +1} bits.

        ASCII mode  : 8 bits/char, MSB first, printable chars only.
        Random mode : 16 bits/char (unsigned, MSB first).

        Returns
        -------
        bits : (n_bits,) float32 array in {-1, +1}
        """
        bits = []
        if self.mode == "ascii":
            for ch in text:
                code = ord(ch)
                if not (32 <= code <= 126):
                    raise ValueError(
                        f"Character '{ch}' (code {code}) is not printable ASCII (32-126). "
                        f"Use random mode for non-printable characters."
                    )
                for shift in range(7, -1, -1):          # MSB first
                    bits.append(1.0 if (code >> shift) & 1 else -1.0)
        else:  # random mode: 16 bits per char
            for ch in text:
                code = ord(ch)
                for shift in range(15, -1, -1):
                    bits.append(1.0 if (code >> shift) & 1 else -1.0)
        return np.array(bits, dtype=np.float32)

    def bits_to_text(self, bits: np.ndarray) -> str:
        """
        Convert a flat {-1, +1} bit array back to text.

        ASCII mode  : 8 bits → one char.
        Random mode : 16 bits → one char.
        """
        binary = (np.sign(bits) > 0).astype(int)
        chars = []
        step = self.bits_per_char
        for i in range(0, len(binary) - step + 1, step):
            byte = binary[i : i + step]
            code = sum(int(b) << (step - 1 - j) for j, b in enumerate(byte))
            if self.mode == "ascii":
                chars.append(chr(code) if 32 <= code <= 126 else "?")
            else:
                try:
                    chars.append(chr(code) if code > 0 else "?")
                except (ValueError, OverflowError):
                    chars.append("?")
        return "".join(chars)

    # ── Chunking helpers ──────────────────────────────────────────────────────

    def _text_to_blocks(self, text: str):
        """
        Split text into padded {-1,+1} blocks of shape (n_blocks, msg_size).

        Padding character is space (ASCII 32) for ASCII mode, or null for random.
        Returns (blocks, original_char_count).
        """
        pad_char = " " if self.mode == "ascii" else "\x00"
        cpb = self.chars_per_block

        # Pad to multiple of chars_per_block
        remainder = len(text) % cpb
        if remainder != 0:
            text += pad_char * (cpb - remainder)

        blocks = []
        for i in range(0, len(text), cpb):
            chunk = text[i : i + cpb]
            bits  = self.text_to_bits(chunk)   # (msg_size,)
            blocks.append(bits)

        return np.array(blocks, dtype=np.float32)   # (n_blocks, msg_size)

    def _blocks_to_text(self, blocks: np.ndarray, original_length: int) -> str:
        """
        Convert (n_blocks, msg_size) array back to text, trimmed to original_length.
        """
        parts = []
        for block in blocks:
            parts.append(self.bits_to_text(block))
        full = "".join(parts)
        return full[:original_length]

    # ── Core encrypt / decrypt ────────────────────────────────────────────────

    def encrypt_text(
        self,
        plaintext: str,
        key: np.ndarray = None,
        output_path: str = None,
    ) -> dict:
        """
        Encrypt a text message using Alice.

        Args:
            plaintext   : The message to encrypt.
            key         : {-1,+1} float32 key array; randomly generated if None.
            output_path : If given, save encrypted data as .npz.

        Returns
        -------
        result : dict with keys
            ciphertext      (n_blocks, 16) float32
            key             (16,) float32
            original_length int
            mode            str
            msg_size        int
        """
        if key is None:
            key = self.generate_key(self.key_size)
            print(f"[Encrypt] Generated key: {key.astype(int).tolist()}")

        original_length = len(plaintext)
        blocks = self._text_to_blocks(plaintext)       # (n_blocks, msg_size)
        n_blocks  = blocks.shape[0]
        key_batch = np.tile(key[np.newaxis, :], (n_blocks, 1)).astype(np.float32)

        t0 = time.time()
        cipher_tf = self.alice(
            tf.concat([tf.constant(blocks), tf.constant(key_batch)], axis=1)
        )
        ciphertext = cipher_tf.numpy()
        t_enc = time.time() - t0

        print(f"[Encrypt] '{plaintext[:40]}{'...' if len(plaintext)>40 else ''}'"
              f"  →  {n_blocks} block(s)  ({t_enc*1000:.1f} ms)")

        result = {
            "ciphertext":      ciphertext.astype(np.float32),
            "key":             key.astype(np.float32),
            "original_length": original_length,
            "mode":            self.mode,
            "msg_size":        self.msg_size,
        }

        if output_path is not None:
            np.savez(output_path, **result)
            print(f"[Encrypt] Saved → {output_path}")

        return result

    def decrypt_text(
        self,
        ciphertext,
        key: np.ndarray,
        original_length: int = None,
        output_path: str = None,
    ) -> str:
        """
        Decrypt ciphertext using Bob.

        Args:
            ciphertext      : (n_blocks, 16) float32 array,
                              OR path to .npz file produced by encrypt_text.
            key             : {-1,+1} float32 key array.
            original_length : If known, trim padding. Auto-detected from .npz.
            output_path     : If given, write decrypted text to this file.

        Returns
        -------
        plaintext : str
        """
        # Load from file if given a path
        if isinstance(ciphertext, (str, os.PathLike)):
            data = np.load(ciphertext, allow_pickle=True)
            ciphertext      = data["ciphertext"]
            original_length = int(data["original_length"])

        ciphertext = np.array(ciphertext, dtype=np.float32)
        n_blocks   = ciphertext.shape[0]
        key_batch  = np.tile(key[np.newaxis, :], (n_blocks, 1)).astype(np.float32)

        t0 = time.time()
        dec_tf = self.bob(
            tf.concat([tf.constant(ciphertext), tf.constant(key_batch)], axis=1)
        )
        dec = dec_tf.numpy()
        t_dec = time.time() - t0

        # Infer original length from blocks if not provided
        if original_length is None:
            original_length = n_blocks * self.chars_per_block

        plaintext = self._blocks_to_text(dec, original_length)
        print(f"[Decrypt] {n_blocks} block(s) → '{plaintext[:40]}"
              f"{'...' if len(plaintext)>40 else ''}'  ({t_dec*1000:.1f} ms)")

        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(plaintext)
            print(f"[Decrypt] Saved → {output_path}")

        return plaintext

    # ── File encrypt / decrypt ────────────────────────────────────────────────

    def encrypt_file(
        self,
        input_path: str,
        output_path: str,
        key: np.ndarray = None,
    ) -> np.ndarray:
        """
        Read a text file and encrypt its contents.

        Returns the key used (useful when auto-generated).
        """
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"[File] Read '{input_path}'  ({len(text)} chars)")

        result = self.encrypt_text(text, key=key, output_path=output_path)
        return result["key"]

    def decrypt_file(
        self,
        input_path: str,
        output_path: str,
        key: np.ndarray,
    ) -> str:
        """
        Decrypt a .npz file and write the plaintext to output_path.
        """
        plaintext = self.decrypt_text(input_path, key, output_path=output_path)
        return plaintext

    # ── Quick security check ──────────────────────────────────────────────────

    def quick_security_check(
        self,
        plaintext: str,
        key: np.ndarray,
        eve_steps: int = 1000,
    ) -> dict:
        """
        Encrypt plaintext, then train a fresh Eve and report her best error.

        Returns dict with keys: best_err, random_baseline, secure (bool).
        """
        from main_enhanced import AttackerNet, l1_distance, eve_loss
        from tensorflow import keras

        result = self.encrypt_text(plaintext, key=key)
        blocks = self._text_to_blocks(plaintext)

        cipher = result["ciphertext"]
        msg_size = self.msg_size

        fresh_eve = AttackerNet(msg_size, False, name="q_eve")
        opt_eve   = keras.optimizers.Adam(0.0008)
        _ = fresh_eve(tf.zeros([1, msg_size]))

        best_err = float("inf")
        BATCH    = min(256, len(cipher))

        for _ in range(eve_steps):
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

        random_baseline = msg_size / 2.0
        secure = best_err >= random_baseline * 0.85
        return {"best_err": best_err, "random_baseline": random_baseline, "secure": secure}
