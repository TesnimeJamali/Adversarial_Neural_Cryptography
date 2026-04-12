"""
=============================================================================
  Image Encryption Tool — Adversarial Neural Cryptography
=============================================================================
  Uses trained Alice/Bob models from main_enhanced.py to encrypt/decrypt
  grayscale images block-by-block (16-bit blocks = 4×4 pixel patches).

  Key improvements in this version
  ──────────────────────────────────
  1. SOFT DECODING  (_blocks_to_image)
     Instead of hard np.sign() threshold, the network's continuous output
     in (−1, +1) is converted to a probability via sigmoid, then multiplied
     by the per-block local median to reconstruct smooth grayscale pixels.
     This preserves local brightness and is robust to small network errors.

  2. PER-BLOCK LOCAL MEDIAN THRESHOLDING  (_image_to_blocks)
     Each 4×4 patch is binarised against its own median value rather than
     a fixed global threshold of 128.  Stored in .npz, reused at decrypt.

  3. VERIFY ENCRYPTION  (verify_encryption)
     Encrypts, decrypts, compares original vs recovered at per-block level,
     prints statistics, and saves a colour-coded error heatmap.

  NOTE: RESEARCH DEMONSTRATION — not production-grade security.
=============================================================================
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_enhanced import CipherNet, AttackerNet


class ImageEncryptor:
    """
    Encrypt and decrypt grayscale images using trained neural cipher models.

    Block size : 4×4 pixels = 16 bits (matches msg_size = 16).
    Binarisation: per-block LOCAL MEDIAN threshold (adapts to brightness).
    Decryption  : SOFT DECODING via sigmoid confidence weighting.
    Key usage   : same 16-bit key for every block (ECB-mode analogue).
    """

    BLOCK_SIZE = 4      # 4×4 pixels = 16 bits

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints/image_patches_4x4",
        msg_size: int = 16,
        key_size: int = 16,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.msg_size = msg_size
        self.key_size = key_size
        self._build_and_load_models()

    # ── Model construction ────────────────────────────────────────────────────

    def _build_and_load_models(self):
        N = self.msg_size + self.key_size
        self.alice = CipherNet(N, self.msg_size, use_attention=False, name="alice")
        self.bob   = CipherNet(N, self.msg_size, use_attention=False, name="bob")
        self.eve   = AttackerNet(self.msg_size,  use_attention=False, name="eve")

        # Warm-up forward pass — required before load_weights with subclassed models
        dummy_msg = tf.zeros([1, self.msg_size])
        dummy_key = tf.zeros([1, self.key_size])
        dummy_c   = self.alice(tf.concat([dummy_msg, dummy_key], axis=1))
        _         = self.bob(tf.concat([dummy_c, dummy_key], axis=1))
        _         = self.eve(dummy_c)

        for model, name in [(self.alice, "alice"), (self.bob, "bob"), (self.eve, "eve")]:
            path = os.path.join(self.checkpoint_dir, f"{name}_final.weights.h5")
            if os.path.exists(path):
                model.load_weights(path)
                print(f"[Load] {name} ← {path}")
            else:
                print(f"[WARNING] Checkpoint not found: {path}")
                print(f"          Using random weights — results will be garbage.")

    # ── Key utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def generate_key(key_size: int = 16) -> np.ndarray:
        return np.random.choice([-1.0, 1.0], size=key_size).astype(np.float32)

    @staticmethod
    def save_key(key: np.ndarray, path: str):
        np.savetxt(path, key[np.newaxis, :], delimiter=",", fmt="%.0f")
        print(f"[Key] Saved → {path}")

    @staticmethod
    def load_key(path: str) -> np.ndarray:
        key = np.loadtxt(path, delimiter=",").astype(np.float32)
        return key.flatten() if key.ndim > 1 else key

    # ── Image ↔ blocks conversion ─────────────────────────────────────────────

    def _image_to_blocks(self, img: np.ndarray):
        """
        Convert a 2D grayscale image to binary {−1, +1} blocks.

        Uses PER-BLOCK LOCAL MEDIAN THRESHOLDING so each 4×4 patch is
        binarised against its own median pixel value, adapting to local
        brightness rather than a fixed global threshold.

        Returns
        -------
        blocks         : (n_blocks, 16) float32 in {−1, +1}
        padded_shape   : (H_pad, W_pad)
        original_shape : (H, W)
        thresholds     : (n_blocks,)  float32  per-block median values
        """
        H, W = img.shape
        bs   = self.BLOCK_SIZE

        # Pad to nearest multiple of block_size with edge-replication
        H_pad = int(np.ceil(H / bs)) * bs
        W_pad = int(np.ceil(W / bs)) * bs
        padded = np.zeros((H_pad, W_pad), dtype=np.float32)
        padded[:H, :W] = img.astype(np.float32)
        if W_pad > W:
            padded[:H, W:] = img[:, -1:].astype(np.float32)
        if H_pad > H:
            padded[H:, :] = padded[H - 1:H, :]

        n_blocks_h = H_pad // bs
        n_blocks_w = W_pad // bs

        # Reshape into flat (n_blocks, 16) patches
        flat_blocks = (
            padded
            .reshape(n_blocks_h, bs, n_blocks_w, bs)
            .transpose(0, 2, 1, 3)
            .reshape(-1, bs * bs)          # (n_blocks, 16)
        )

        # Per-block median → threshold
        thresholds = np.median(flat_blocks, axis=1)            # (n_blocks,)

        # Binarise: pixel > block_median → +1, else −1
        binary = np.where(
            flat_blocks > thresholds[:, np.newaxis], 1.0, -1.0
        ).astype(np.float32)

        return binary, (H_pad, W_pad), (H, W), thresholds.astype(np.float32)

    def _blocks_to_image(
        self,
        blocks: np.ndarray,
        padded_shape: tuple,
        original_shape: tuple,
        thresholds: np.ndarray = None,
    ) -> np.ndarray:
        """
        Convert decrypted network outputs to a grayscale image using SOFT DECODING.

        Why soft decoding?
        ------------------
        Hard threshold (np.sign) maps any output, however uncertain, to ±1.
        A barely-positive 0.02 and a confident 0.98 both become +1, losing
        the network's uncertainty estimate.  On slightly wrong bits this
        produces harsh black/white artefacts instead of graceful degradation.

        Soft decoding converts the tanh output ∈ (−1, +1) → probability
        p = sigmoid(output × 4) ∈ (0, 1), then reconstructs:

            pixel = local_threshold × p

        So:
            confident +1 (p ≈ 1) → pixel ≈ threshold  (original brightness)
            confident -1 (p ≈ 0) → pixel ≈ 0
            uncertain bit (p ≈ 0.5) → pixel ≈ threshold / 2  (mid-grey)

        The ×4 sharpening factor makes the sigmoid steep enough that
        high-confidence bits still converge close to 0 or threshold while
        uncertain bits produce a smooth mid-grey rather than hard artefacts.

        Parameters
        ----------
        blocks         : (n_blocks, 16) raw network output in (−1, +1)
        padded_shape   : (H_pad, W_pad)
        original_shape : (H, W)
        thresholds     : (n_blocks,) per-block median from encryption.
                         Falls back to hard {0, 255} if None.
        """
        H_pad, W_pad   = padded_shape
        H_orig, W_orig = original_shape
        bs = self.BLOCK_SIZE
        n_blocks_h = H_pad // bs
        n_blocks_w = W_pad // bs

        if thresholds is not None and len(thresholds) == blocks.shape[0]:
            # ── Soft decoding ──────────────────────────────────────────────
            # Scale tanh output to sharpen the sigmoid boundary:
            #   ×4 means output=0.5 → p≈0.88 (confident positive)
            #         output=-0.5 → p≈0.12 (confident negative)
            logits = blocks * 4.0                              # (n_blocks, 16)
            probs  = 1.0 / (1.0 + np.exp(-logits))            # sigmoid → (0,1)

            # Reconstruct pixel as threshold × probability
            thresh_col = thresholds[:, np.newaxis]             # (n_blocks, 1)
            pixel_vals = thresh_col * probs                    # (n_blocks, 16)

        else:
            # ── Fallback: hard threshold → {0, 255} ───────────────────────
            bits       = np.sign(blocks)
            pixel_vals = np.where(bits > 0, 255.0, 0.0)

        # Reassemble blocks → padded image
        img_pad = (
            pixel_vals
            .reshape(n_blocks_h, n_blocks_w, bs, bs)
            .transpose(0, 2, 1, 3)
            .reshape(H_pad, W_pad)
        )

        img_uint8 = np.clip(img_pad, 0, 255).astype(np.uint8)
        return img_uint8[:H_orig, :W_orig]

    # ── Error tracking ─────────────────────────────────────────────────────────

    def verify_encryption(
        self,
        image_path: str,
        key: np.ndarray = None,
        output_heatmap: str = "error_heatmap.png",
    ) -> dict:
        """
        Encrypt then decrypt an image and report per-block error statistics.

        For each 4×4 block, counts how many of the 16 bits were incorrectly
        recovered by Bob, then saves a colour-coded heatmap:
            green  = 0 bit errors (perfect block)
            yellow = 1–4 bit errors (minor)
            red    = 5+ bit errors (major)

        Parameters
        ----------
        image_path     : input image path (any PIL-readable format)
        key            : {−1,+1} float32 key; randomly generated if None
        output_heatmap : PNG output path (pass None to skip saving)

        Returns
        -------
        stats : dict with keys
            total_blocks, total_bits, total_errors,
            mean_errors_per_block, blocks_perfect, blocks_minor,
            blocks_major, worst_block_idx, worst_block_errors, psnr
        """
        import matplotlib.pyplot as plt
        from PIL import Image

        if key is None:
            key = self.generate_key(self.key_size)
            print(f"[Verify] Generated random key: {key.astype(int).tolist()}")

        img_np = np.array(Image.open(image_path).convert("L"), dtype=np.uint8)
        H, W   = img_np.shape
        blocks_orig, padded_shape, orig_shape, thresholds = self._image_to_blocks(img_np)
        n_blocks  = blocks_orig.shape[0]
        key_batch = np.tile(key[np.newaxis, :], (n_blocks, 1)).astype(np.float32)
        BATCH     = 4096

        # ── Alice: plaintext → ciphertext ─────────────────────────────────
        ciphertexts = []
        for s in range(0, n_blocks, BATCH):
            e = min(s + BATCH, n_blocks)
            c = self.alice(tf.concat([
                tf.constant(blocks_orig[s:e]),
                tf.constant(key_batch[s:e]),
            ], axis=1))
            ciphertexts.append(c.numpy())
        cipher = np.concatenate(ciphertexts, axis=0)      # (n_blocks, 16)

        # ── Bob: ciphertext → decrypted output ────────────────────────────
        dec_list = []
        for s in range(0, n_blocks, BATCH):
            e = min(s + BATCH, n_blocks)
            d = self.bob(tf.concat([
                tf.constant(cipher[s:e]),
                tf.constant(key_batch[s:e]),
            ], axis=1))
            dec_list.append(d.numpy())
        dec = np.concatenate(dec_list, axis=0)             # (n_blocks, 16)

        # ── Per-block bit error count ──────────────────────────────────────
        # Hard-threshold for error counting (binary comparison)
        recovered_bits   = np.sign(dec)                    # (n_blocks, 16)
        errors_per_block = np.sum(
            recovered_bits != blocks_orig, axis=1
        ).astype(np.float32)                               # (n_blocks,)

        total_errors   = int(errors_per_block.sum())
        mean_errors    = float(errors_per_block.mean())
        worst_idx      = int(errors_per_block.argmax())
        worst_errs     = int(errors_per_block[worst_idx])
        blocks_perfect = int((errors_per_block == 0).sum())
        blocks_minor   = int(((errors_per_block >= 1) & (errors_per_block <= 4)).sum())
        blocks_major   = int((errors_per_block > 4).sum())

        # PSNR on the soft-decoded reconstruction
        img_dec = self._blocks_to_image(dec, padded_shape, orig_shape, thresholds)
        mse  = np.mean((img_np.astype(float) - img_dec.astype(float)) ** 2)
        psnr = float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)

        # ── Print summary ──────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  ENCRYPTION VERIFICATION — {os.path.basename(image_path)}")
        print(f"{'='*60}")
        print(f"  Image size        : {H}×{W} px")
        print(f"  Total blocks      : {n_blocks}  (4×4 px each)")
        print(f"  Total bits        : {n_blocks * self.msg_size}")
        print(f"  Total bit errors  : {total_errors}")
        print(f"  Mean errors/block : {mean_errors:.3f} / {self.msg_size} bits")
        print(f"  Perfect blocks    : {blocks_perfect}  ({100*blocks_perfect/n_blocks:.1f}%)")
        print(f"  Minor  (1–4 err)  : {blocks_minor}  ({100*blocks_minor/n_blocks:.1f}%)")
        print(f"  Major  (5+ err)   : {blocks_major}  ({100*blocks_major/n_blocks:.1f}%)")
        print(f"  Worst block idx   : {worst_idx}  ({worst_errs} errors)")
        print(f"  PSNR (soft decode): {psnr:.2f} dB")
        print(f"{'='*60}\n")

        # ── Heatmap ────────────────────────────────────────────────────────
        if output_heatmap is not None:
            H_pad, W_pad = padded_shape
            bs           = self.BLOCK_SIZE
            n_h = H_pad // bs
            n_w = W_pad // bs

            # Reshape error counts into a 2-D block grid
            err_grid   = errors_per_block.reshape(n_h, n_w)
            # Upscale to pixel resolution for display
            err_pixels = np.repeat(np.repeat(err_grid, bs, axis=0), bs, axis=1)
            err_pixels = err_pixels[:H, :W]

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(img_np,  cmap="gray", vmin=0, vmax=255)
            axes[0].set_title(f"Original ({H}×{W} px)", fontsize=11)
            axes[0].axis("off")

            axes[1].imshow(img_dec, cmap="gray", vmin=0, vmax=255)
            axes[1].set_title(
                f"Decrypted (soft decode)\n"
                f"PSNR = {psnr:.1f} dB  |  Mean Bob error = {mean_errors:.2f} bits",
                fontsize=10,
            )
            axes[1].axis("off")

            # Heatmap: 0 errors = green, many errors = red
            hm = axes[2].imshow(
                err_pixels,
                cmap="RdYlGn_r",        # green=good, red=bad
                vmin=0,
                vmax=self.msg_size,     # full scale 0–16
                interpolation="nearest",
            )
            plt.colorbar(hm, ax=axes[2], label="Bit errors per 4×4 block")
            axes[2].set_title(
                f"Error Heatmap\n"
                f"Green=perfect  Yellow=1–4 errors  Red=5+ errors",
                fontsize=10,
            )
            axes[2].axis("off")

            # Annotate block-error counts on small images
            if n_h * n_w <= 400:
                for r in range(n_h):
                    for c in range(n_w):
                        e_val  = int(err_grid[r, c])
                        colour = "white" if e_val > self.msg_size // 2 else "black"
                        axes[2].text(
                            c * bs + bs / 2,
                            r * bs + bs / 2,
                            str(e_val),
                            ha="center", va="center",
                            fontsize=6, color=colour,
                        )

            fig.suptitle(
                "Neural Cipher — Per-Block Bit-Error Heatmap\n"
                "[RESEARCH DEMO — NOT production-grade security]",
                fontsize=12, color="darkred",
            )
            plt.tight_layout()
            plt.savefig(output_heatmap, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[Verify] Heatmap → {output_heatmap}")

        return {
            "total_blocks":          n_blocks,
            "total_bits":            n_blocks * self.msg_size,
            "total_errors":          total_errors,
            "mean_errors_per_block": mean_errors,
            "blocks_perfect":        blocks_perfect,
            "blocks_minor":          blocks_minor,
            "blocks_major":          blocks_major,
            "worst_block_idx":       worst_idx,
            "worst_block_errors":    worst_errs,
            "psnr":                  psnr,
        }

    # ── Core encrypt / decrypt ─────────────────────────────────────────────────

    def encrypt_image(
        self,
        image_path: str,
        key: np.ndarray,
        output_path: str = None,
    ) -> np.ndarray:
        """
        Encrypt a grayscale image using Alice.

        Saves a .npz containing ciphertext, original/padded shapes, block_size,
        key, and per-block median thresholds (needed for soft decoding).
        """
        from PIL import Image

        img = np.array(Image.open(image_path).convert("L"), dtype=np.uint8)
        H, W = img.shape
        print(f"[Encrypt] Image: {image_path}  ({H}×{W})")

        blocks, padded_shape, original_shape, thresholds = self._image_to_blocks(img)
        n_blocks  = blocks.shape[0]
        key_batch = np.tile(key[np.newaxis, :], (n_blocks, 1)).astype(np.float32)
        print(f"[Encrypt] {n_blocks} blocks  ({self.BLOCK_SIZE}×{self.BLOCK_SIZE} px each)")

        BATCH = 4096
        ciphertexts = []
        t0 = time.time()
        for start in range(0, n_blocks, BATCH):
            end   = min(start + BATCH, n_blocks)
            b_msg = tf.constant(blocks[start:end])
            b_key = tf.constant(key_batch[start:end])
            c     = self.alice(tf.concat([b_msg, b_key], axis=1))
            ciphertexts.append(c.numpy())
        encrypted_blocks = np.concatenate(ciphertexts, axis=0)
        print(f"[Encrypt] Done in {time.time()-t0:.3f}s")

        if output_path is None:
            output_path = os.path.splitext(image_path)[0] + "_encrypted.npz"

        np.savez(
            output_path,
            ciphertext     = encrypted_blocks.astype(np.float32),
            original_shape = np.array(original_shape),
            padded_shape   = np.array(padded_shape),
            block_size     = np.array(self.BLOCK_SIZE),
            key            = key.astype(np.float32),
            thresholds     = thresholds.astype(np.float32),
        )
        print(f"[Encrypt] Saved → {output_path}")
        return encrypted_blocks

    def decrypt_image(
        self,
        encrypted_path: str,
        key: np.ndarray,
        output_path: str = None,
    ) -> np.ndarray:
        """
        Decrypt a .npz encrypted image using Bob with SOFT DECODING.

        Loads the per-block thresholds stored at encryption time and passes
        them to _blocks_to_image for brightness-aware reconstruction.
        """
        from PIL import Image

        data           = np.load(encrypted_path)
        ciphertext     = data["ciphertext"]
        original_shape = tuple(data["original_shape"])
        padded_shape   = tuple(data["padded_shape"])
        thresholds     = data["thresholds"] if "thresholds" in data else None
        n_blocks       = ciphertext.shape[0]
        print(f"[Decrypt] {encrypted_path}  ({n_blocks} blocks)")

        key_batch = np.tile(key[np.newaxis, :], (n_blocks, 1)).astype(np.float32)

        BATCH = 4096
        dec_blocks = []
        t0 = time.time()
        for start in range(0, n_blocks, BATCH):
            end   = min(start + BATCH, n_blocks)
            b_c   = tf.constant(ciphertext[start:end])
            b_key = tf.constant(key_batch[start:end])
            d     = self.bob(tf.concat([b_c, b_key], axis=1))
            dec_blocks.append(d.numpy())
        decrypted = np.concatenate(dec_blocks, axis=0)
        print(f"[Decrypt] Done in {time.time()-t0:.3f}s")

        img = self._blocks_to_image(decrypted, padded_shape, original_shape, thresholds)

        if output_path is None:
            output_path = os.path.splitext(encrypted_path)[0] + "_decrypted.png"

        Image.fromarray(img).save(output_path)
        print(f"[Decrypt] Saved → {output_path}")
        return img

    # ── Batch processing ──────────────────────────────────────────────────────

    def encrypt_batch(self, input_dir: str, output_dir: str, key: np.ndarray):
        SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif"}
        os.makedirs(output_dir, exist_ok=True)
        paths = [p for p in Path(input_dir).iterdir()
                 if p.suffix.lower() in SUPPORTED]
        if not paths:
            print(f"[Batch] No images found in {input_dir}"); return
        print(f"[Batch] Encrypting {len(paths)} images …")
        for p in sorted(paths):
            out = os.path.join(output_dir, p.stem + "_encrypted.npz")
            try:
                self.encrypt_image(str(p), key, output_path=out)
            except Exception as e:
                print(f"[Batch] ERROR on {p.name}: {e}")

    def decrypt_batch(self, input_dir: str, output_dir: str, key: np.ndarray):
        os.makedirs(output_dir, exist_ok=True)
        paths = list(Path(input_dir).glob("*.npz"))
        if not paths:
            print(f"[Batch] No .npz files in {input_dir}"); return
        print(f"[Batch] Decrypting {len(paths)} files …")
        for p in sorted(paths):
            out = os.path.join(output_dir,
                               p.stem.replace("_encrypted", "") + "_decrypted.png")
            try:
                self.decrypt_image(str(p), key, output_path=out)
            except Exception as e:
                print(f"[Batch] ERROR on {p.name}: {e}")

    # ── Cipher visualisation ──────────────────────────────────────────────────

    def visualize_ciphertext(self, encrypted_path: str) -> np.ndarray:
        """Map ciphertext [−1,+1] → [0,255] for visual inspection (not decryption)."""
        data           = np.load(encrypted_path)
        ciphertext     = data["ciphertext"]
        original_shape = tuple(data["original_shape"])
        padded_shape   = tuple(data["padded_shape"])
        bs             = int(data["block_size"])

        H_pad, W_pad   = padded_shape
        H_orig, W_orig = original_shape
        n_blocks_h = H_pad // bs
        n_blocks_w = W_pad // bs

        vis = ((ciphertext + 1.0) / 2.0 * 255.0).astype(np.uint8)
        vis_img = (
            vis
            .reshape(n_blocks_h, n_blocks_w, bs, bs)
            .transpose(0, 2, 1, 3)
            .reshape(H_pad, W_pad)
        )
        return vis_img[:H_orig, :W_orig]
