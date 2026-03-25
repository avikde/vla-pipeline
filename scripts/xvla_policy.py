"""
X-VLA policy wrapper: loading, tokenization, preprocessing, inference.

Robot utilities (EE state, rotation math, action decoding, reference trajectory)
live in widowx_control.py.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_policy(checkpoint: str = "lerobot/xvla-widowx", device: str | None = None):
    """Load XVLAPolicy and its BART tokenizer. Prints config and validates action_mode.

    Returns (policy, tokenizer, device).
    Raises ImportError if lerobot[xvla] is not installed.
    """
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from transformers import AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = XVLAPolicy.from_pretrained(checkpoint).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(policy.config.tokenizer_name)

    cfg = policy.config
    print(f"  ✓ Device: {device}")
    print(f"\n  📋 Policy Configuration:")
    print(f"     - action_mode:         {cfg.action_mode}")
    print(f"     - chunk_size:          {cfg.chunk_size}")
    print(f"     - n_action_steps:      {cfg.n_action_steps}")
    print(f"     - num_denoising_steps: {cfg.num_denoising_steps}")
    print(f"     - use_proprio:         {cfg.use_proprio}")
    print(f"     - max_action_dim:      {cfg.max_action_dim}")

    if cfg.action_mode != "ee6d":
        print(f"\n  ⚠️  WARNING: action_mode is '{cfg.action_mode}', expected 'ee6d'")
        print(f"     X-VLA WidowX is trained with EE actions, not joint actions!")
    else:
        print(f"  ✓ Confirmed: Using EE (end-effector) action mode")

    return policy, tokenizer, device


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------

_token_cache: dict = {}


def tokenize_task(task: str, tokenizer, policy_config, device: str) -> tuple:
    """Tokenize a task string. Results are cached by task string.

    Returns (input_ids, attention_mask) as tensors on device.
    """
    if task not in _token_cache:
        tok = tokenizer(
            task,
            padding="max_length",
            max_length=policy_config.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        _token_cache[task] = (
            tok["input_ids"].to(device),
            tok["attention_mask"].to(device),
        )
    return _token_cache[task]


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(rgb_array: np.ndarray, device: str) -> torch.Tensor:
    """Convert H×W×3 uint8 numpy array to (1, 3, H, W) float tensor on device."""
    tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Observation building
# ---------------------------------------------------------------------------

def build_observation(
    img: np.ndarray,
    img2: np.ndarray,
    ee_state_8d: np.ndarray,
    language_tokens: torch.Tensor,
    language_attention_mask: torch.Tensor,
    device: str,
) -> dict:
    """Pack images + proprio + language into the dict XVLAPolicy expects."""
    return {
        "observation.images.image":  preprocess_image(img, device),
        "observation.images.image2": preprocess_image(img2, device),
        "observation.state": torch.from_numpy(ee_state_8d).float().unsqueeze(0).to(device),
        "observation.language.tokens": language_tokens,
        "observation.language.attention_mask": language_attention_mask,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def select_action(policy, observation: dict, device: str) -> np.ndarray:
    """Run one policy step. Returns flat float32 numpy action vector."""
    with torch.inference_mode():
        try:
            actions = policy.select_action(observation)
        except Exception as e:
            print(f"[xvla_policy] inference error: {e}")
            actions = torch.zeros(20, device=device)

    if device == "cuda":
        torch.cuda.synchronize()

    if isinstance(actions, torch.Tensor):
        return actions.detach().cpu().numpy().flatten()
    return np.array(actions).flatten()


