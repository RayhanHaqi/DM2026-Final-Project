"""Cache and blend class-probability predictions for ordinal severity scores."""

import os

import numpy as np
import pandas as pd

from model import experiments
from model.ordinal_tree import expected_values_from_week_class_probs


def soft_probs_from_regression(values, temperature=None):
    """Map regression values in [0, 5] to class probabilities over scores 0..5.

    Default mode linearly interpolates between floor and ceil scores so the
    expected value matches the input prediction. Optional temperature spreads
    mass to neighboring scores (Gaussian bump); use only when exploring,
    because it shifts means.
    """
    arr = np.clip(np.asarray(values, dtype=float), 0.0, 5.0)
    if arr.ndim == 1:
        arr = arr[:, None]

    if temperature is None:
        low = np.floor(arr).astype(int)
        high = np.ceil(arr).astype(int)
        frac = arr - low
        probs = np.zeros(arr.shape + (6,), dtype=float)
        for class_id in range(6):
            probs[..., class_id] = np.where(
                low == class_id,
                1.0 - frac,
                np.where(high == class_id, frac, 0.0),
            )
        return probs

    score_ids = np.arange(6, dtype=float)
    diff = score_ids.reshape(1, 1, 6) - arr[..., None]
    logits = -0.5 * (diff / max(float(temperature), 1e-6)) ** 2
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    return np.clip(probs, 0.0, 1.0)


def blend_class_probs(prob_list, weights):
    """Weighted average of class-probability tensors with shape (n_samples, n_weeks, 6)."""
    if len(prob_list) != len(weights):
        raise ValueError("prob_list and weights must have the same length")
    if not prob_list:
        raise ValueError("prob_list must not be empty")

    weight_arr = np.asarray(weights, dtype=float)
    if np.any(weight_arr < 0):
        raise ValueError("weights must be non-negative")
    if weight_arr.sum() <= 0:
        raise ValueError("weights must sum to a positive value")
    weight_arr = weight_arr / weight_arr.sum()

    base_shape = prob_list[0].shape
    for probs in prob_list:
        if probs.shape != base_shape:
            raise ValueError("all probability caches must have the same shape")

    blended = np.zeros(base_shape, dtype=float)
    for weight, probs in zip(weight_arr, prob_list):
        blended += weight * probs

    row_sums = blended.sum(axis=-1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return np.clip(blended / row_sums, 0.0, 1.0)


def posterior_quantile_from_class_probs(class_probs, quantile=0.5):
    """Map class probabilities to a score via linear interpolation on scores 0..5.

    quantile=0.5 is the posterior median, which minimizes MAE when Y is discrete on
    {0,..,5} and the action may be continuous in [0, 5].
    """
    arr = np.clip(np.asarray(class_probs, dtype=float), 0.0, 1.0)
    if arr.ndim != 3 or arr.shape[-1] != 6:
        raise ValueError("class_probs must have shape (n_samples, n_weeks, 6)")

    row_sums = arr.sum(axis=-1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    arr = arr / row_sums

    q = float(np.clip(quantile, 0.0, 1.0))
    flat = arr.reshape(-1, 6)
    scores = np.arange(6, dtype=float)
    out = np.empty(flat.shape[0], dtype=float)

    for row_idx, probs in enumerate(flat):
        cdf = probs.cumsum()
        if q <= 0.0:
            out[row_idx] = 0.0
            continue
        if q >= 1.0:
            out[row_idx] = 5.0
            continue

        idx = int(np.searchsorted(cdf, q, side="left"))
        idx = min(max(idx, 0), 5)
        if idx == 0:
            out[row_idx] = 0.0
        elif cdf[idx] <= cdf[idx - 1] + 1e-12:
            out[row_idx] = scores[idx]
        else:
            frac = (q - cdf[idx - 1]) / (cdf[idx] - cdf[idx - 1])
            out[row_idx] = scores[idx - 1] + frac * (scores[idx] - scores[idx - 1])

    return np.clip(out.reshape(arr.shape[0], arr.shape[1]), 0.0, 5.0)


def summarize_class_prob_mass(class_probs):
    """Mean class masses and P(Y>=k) averaged over regions and weeks."""
    arr = np.asarray(class_probs, dtype=float)
    if arr.ndim != 3 or arr.shape[-1] != 6:
        raise ValueError("class_probs must have shape (n_samples, n_weeks, 6)")

    row_sums = arr.sum(axis=-1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    normed = arr / row_sums
    mean_mass = normed.mean(axis=(0, 1))
    cum = np.cumsum(mean_mass)
    survival = {f"p_ge_{k}": float(1.0 - cum[k - 1]) for k in range(1, 6)}
    return {
        "mean_class_mass": {str(i): float(mean_mass[i]) for i in range(6)},
        **survival,
    }


def decoder_global_mean(class_probs, decision="mean", quantile=0.5):
    """Global mean of decoded scores (diagnostic only)."""
    preds = class_probs_to_predictions(class_probs, decision=decision, quantile=quantile)
    return float(np.mean(preds))


def diagnose_prob_blend_decoders(cache_specs, target_region_ids, quantiles=(0.48, 0.5, 0.52)):
    """Compare mean vs quantile decoders on each cache and the weighted blend.

    Median/quantile are invalid on soft_probs_from_regression caches (mean-preserving
    encoding). Use this before submitting non-mean decision rules.
    """
    parsed = parse_weighted_cache_specs(cache_specs)
    target_region_ids = [str(rid) for rid in target_region_ids]
    layers = []

    loaded = []
    for path, weight in parsed:
        item = load_prob_cache(path)
        aligned = reorder_class_probs(
            item["class_probs"], item["region_ids"], target_region_ids
        )
        loaded.append((path, weight, item["source"], aligned))
        row = {
            "layer": "cache",
            "path": path,
            "weight": weight,
            "source": item["source"],
            "mass": summarize_class_prob_mass(aligned),
            "decode_mean": decoder_global_mean(aligned, decision="mean"),
            "decode_median": decoder_global_mean(aligned, decision="median"),
        }
        for q in quantiles:
            row[f"decode_q{int(round(q * 100)):02d}"] = decoder_global_mean(
                aligned, decision="quantile", quantile=q
            )
        layers.append(row)

    weights = [weight for _, weight, _, _ in loaded]
    aligned_probs = [aligned for _, _, _, aligned in loaded]
    blended = blend_class_probs(aligned_probs, weights)
    blend_row = {
        "layer": "blend",
        "path": "+".join(f"{os.path.basename(p)}:{w}" for p, w, _, _ in loaded),
        "weight": 1.0,
        "source": "blend",
        "mass": summarize_class_prob_mass(blended),
        "decode_mean": decoder_global_mean(blended, decision="mean"),
        "decode_median": decoder_global_mean(blended, decision="median"),
    }
    for q in quantiles:
        blend_row[f"decode_q{int(round(q * 100)):02d}"] = decoder_global_mean(
            blended, decision="quantile", quantile=q
        )
    layers.append(blend_row)
    return layers


def class_probs_to_predictions(class_probs, decision="mean", quantile=0.5):
    """Convert (n_samples, n_weeks, 6) class probabilities to (n_samples, n_weeks) scores.

    decision:
      - mean: expected value (default; correct for soft_probs_from_regression caches)
      - median: posterior quantile q=0.5 (only valid for calibrated posteriors)
      - quantile: linear quantile on scores 0..5 (only valid for calibrated posteriors)
    """
    if decision == "mean":
        return expected_values_from_week_class_probs(class_probs)
    if decision == "median":
        return posterior_quantile_from_class_probs(class_probs, 0.5)
    if decision == "quantile":
        return posterior_quantile_from_class_probs(class_probs, quantile)
    raise ValueError(f"decision must be mean, median, or quantile; got {decision!r}")


def _encode_metadata_value(value):
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        return np.array(value)
    if value is None:
        return np.array("")
    return np.array(str(value))


def save_prob_cache(path, class_probs, region_ids, source, metadata=None):
    """Save probability cache to .npz (class_probs shape: n_regions x n_weeks x 6).

    region_ids order is whatever the producer used (test groupby or submission CSV).
    Consumers blending multiple caches must pass sample_submission region order and
    call reorder_class_probs before blend_class_probs.
    """
    arr = np.asarray(class_probs, dtype=float)
    if arr.ndim != 3 or arr.shape[-1] != 6:
        raise ValueError("class_probs must have shape (n_regions, n_weeks, 6)")

    payload = {
        "class_probs": arr,
        "region_ids": np.asarray(region_ids),
        "source": np.array(source),
    }
    if metadata:
        for key, value in metadata.items():
            payload[f"meta_{key}"] = _encode_metadata_value(value)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def reorder_class_probs(class_probs, region_ids, target_region_ids):
    """Reorder (n_regions, n_weeks, 6) probabilities to match target_region_ids."""
    lookup = {str(rid): idx for idx, rid in enumerate(region_ids)}
    missing = [rid for rid in target_region_ids if rid not in lookup]
    if missing:
        raise ValueError(
            f"Probability cache is missing {len(missing)} regions; first missing id: {missing[0]}"
        )
    indices = [lookup[str(rid)] for rid in target_region_ids]
    return np.asarray(class_probs, dtype=float)[indices]


def load_prob_cache(path):
    """Load probability cache; returns dict with class_probs, region_ids, source."""
    data = np.load(path, allow_pickle=True)
    source = data["source"]
    if isinstance(source, np.ndarray):
        source = str(source.item()) if source.shape == () else str(source[0])

    result = {
        "class_probs": np.asarray(data["class_probs"], dtype=float),
        "region_ids": [str(rid) for rid in data["region_ids"].tolist()],
        "source": source,
        "path": path,
    }
    metadata = {}
    for key in data.files:
        if key.startswith("meta_"):
            metadata[key[5:]] = data[key]
    if metadata:
        result["metadata"] = metadata
    return result


def parse_weighted_cache_specs(specs):
    """Parse ['path.npz:0.15', 'other.npz:0.85'] into [(path, weight), ...]."""
    parsed = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Cache spec must be path:weight, got {spec!r}")
        path, weight_str = spec.rsplit(":", 1)
        path = path.strip()
        weight = float(weight_str.strip())
        if not path:
            raise ValueError(f"Missing cache path in spec {spec!r}")
        parsed.append((path, weight))
    return parsed


def blend_prob_caches(cache_specs, target_region_ids):
    """Load weighted caches, align regions, and return blended class probabilities."""
    parsed = parse_weighted_cache_specs(cache_specs)
    loaded = [load_prob_cache(path) for path, _ in parsed]
    target_region_ids = [str(rid) for rid in target_region_ids]

    aligned_probs = [
        reorder_class_probs(item["class_probs"], item["region_ids"], target_region_ids)
        for item in loaded
    ]
    weights = [weight for _, weight in parsed]
    blended = blend_class_probs(aligned_probs, weights)
    return blended, target_region_ids, loaded


def write_prob_blend_submission(
    cache_specs,
    sample_path,
    output_path,
    decision="mean",
    quantile=0.5,
):
    """Blend caches and write a submission CSV in sample_submission row order."""
    sample = pd.read_csv(sample_path)
    target_region_ids = sample.iloc[:, 0].tolist()
    blended, region_ids, sources = blend_prob_caches(cache_specs, target_region_ids)
    preds = class_probs_to_predictions(blended, decision=decision, quantile=quantile)
    sub = experiments.build_submission(region_ids, preds, sample)
    ok, messages = experiments.validate_submission(sub, sample)
    if not ok:
        raise ValueError("; ".join(messages))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sub.to_csv(output_path, index=False)
    return output_path, sources
