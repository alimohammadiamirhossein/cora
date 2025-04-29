import torch

def distance_to_similarity(distances, temperature=1.0):
    """
    Turns a distance matrix into a similarity matrix so it works with distribution-based metrics.
    """
    similarities = torch.exp(-distances / temperature)
    similarities = torch.clamp(similarities, min=1e-8)
    return similarities

#################################
##   "New Object" Detection    ##
#################################

def detect_newness_two_sided(distances, k=3, quantile=0.97):
    device = distances.device
    N_src, N_tgt = distances.shape

    topk_src_idx_t = torch.topk(distances, k, dim=0, largest=False).indices  # [k, N_tgt]
    topk_tgt_idx_s = torch.topk(distances, k, dim=1, largest=False).indices  # [N_src, k]

    src_to_tgt_mask = torch.zeros((N_src, N_tgt), device=device)
    tgt_to_src_mask = torch.zeros((N_src, N_tgt), device=device)

    row_indices = topk_src_idx_t  # [k, N_tgt]
    col_indices = torch.arange(N_tgt, device=device).unsqueeze(0).repeat(k, 1)  # [k, N_tgt]
    src_to_tgt_mask[row_indices, col_indices] = 1.0  # Assign 1.0 at the top-k positions

    row_indices = torch.arange(N_src, device=device).unsqueeze(1).repeat(1, k)  # [N_src, k]
    col_indices = topk_tgt_idx_s  # [N_src, k]
    tgt_to_src_mask[row_indices, col_indices] = 1.0  # Assign 1.0 at the top-k positions

    overlap_mask = (src_to_tgt_mask * tgt_to_src_mask).sum(dim=0) > 0  # [N_tgt]

    distances[:, overlap_mask] = 0.0

    two_sided_mask = (~overlap_mask).float()

    min_distances, _ = distances.min(dim=0)
    threshold = torch.quantile(min_distances, quantile)
    threshold_mask = (min_distances > threshold).float()

    combined_mask = two_sided_mask * threshold_mask
    return combined_mask

def detect_newness_distance(min_distances, quantile=0.97):
    """
    Old approach: threshold on min distance at a chosen percentile.
    """
    threshold = torch.quantile(min_distances, quantile)
    newness_mask = (min_distances > threshold).float()
    return newness_mask

def detect_newness_topk_margin(distances, top_k=2, quantile=0.03):
    """
    Top-k margin approach in distance space.
    distances: [N_src, N_tgt]
    Sort each column ascending => best match is index 0, second best is index 1, etc.
    A smaller margin => ambiguous => likely new.
    We threshold the margin at some percentile.
    """
    sorted_dists, _ = torch.sort(distances, dim=0)  
    best = sorted_dists[0]                        # [N_tgt]
    second_best = sorted_dists[1] if top_k >= 2 else sorted_dists[0]  # [N_tgt]
    margin = second_best - best  # [N_tgt]

    # If margin < threshold => ambiguous => "new"
    # We'll pick threshold as a quantile of margin
    threshold = torch.quantile(margin, quantile)
    newness_mask = (margin < threshold).float()
    return newness_mask

def detect_newness_entropy(distances, temperature=1.0, quantile=0.97):
    """
    Entropy-based approach. First convert distance->similarity with an exponential.
    Then normalize to get a distribution for each target patch, compute Shannon entropy.
    High entropy => new object (no strong match).
    """
    similarities = distance_to_similarity(distances, temperature=temperature)
    probs = similarities / similarities.sum(dim=0, keepdim=True)  # [N_src, N_tgt]
    # Shannon Entropy: -sum(p log p)
    entropy = -torch.sum(probs * torch.log(probs), dim=0)  # [N_tgt]

    # threshold
    threshold = torch.quantile(entropy, quantile)
    newness_mask = (entropy > threshold).float()
    return newness_mask

def detect_newness_gini(distances, temperature=1.0, quantile=0.97):
    """
    Gini impurity-based approach. Convert distances to similarities,
    get a distribution, compute Gini.
    High Gini => wide distribution => new object.
    """
    similarities = distance_to_similarity(distances, temperature=temperature)
    probs = similarities / similarities.sum(dim=0, keepdim=True)
    # Gini: sum(p_i*(1-p_i)) => high if spread out
    gini = torch.sum(probs * (1.0 - probs), dim=0)  # [N_tgt]

    threshold = torch.quantile(gini, quantile)
    newness_mask = (gini > threshold).float()
    return newness_mask

def detect_newness_kl(distances, temperature=1.0, quantile=0.97):
    """
    KL-based approach. Compare distribution to uniform => if close to uniform => new object.
    1) Convert distances -> similarities
    2) p(x) = similarities / sum(similarities)
    3) KL(p || uniform) => sum p(x) log (p(x)/(1/N_src))
    4) If p is near uniform => KL small => new object.
       We'll invert it => newness ~ 1/KL.
    """
    similarities = distance_to_similarity(distances, temperature=temperature)
    N_src = distances.shape[0]
    probs = similarities / similarities.sum(dim=0, keepdim=True)

    uniform_val = 1.0 / float(N_src)
    kl_vals = torch.sum(probs * torch.log(probs / uniform_val), dim=0)  # [N_tgt]
    inv_kl = 1.0 / (kl_vals + 1e-8)  # big => distribution is near uniform => new

    threshold = torch.quantile(inv_kl, quantile)
    newness_mask = (inv_kl > threshold).float()
    return newness_mask

def detect_newness_variation_ratio(distances, temperature=1.0, quantile=0.97):
    """
    Variation Ratio: 1 - max(prob).
    1) Convert distance->similarity
    2) p(x) = sim(x) / sum_x'(sim(x'))
    3) var_ratio = 1 - max(p)
    High var_ratio => new object.
    """
    similarities = distance_to_similarity(distances, temperature=temperature)
    probs = similarities / similarities.sum(dim=0, keepdim=True)
    max_prob, _ = torch.max(probs, dim=0)  # [N_tgt]
    var_ratio = 1.0 - max_prob

    threshold = torch.quantile(var_ratio, quantile)
    newness_mask = (var_ratio > threshold).float()
    return newness_mask


def detect_newness_two_sided_ratio(
    distances,
    top_k_ratio_quantile=0.03,
    two_sided=True
):
    """
    Two-sided matching + ratio test in distance space.

    Ratio test: For each t, let d0 = best distance, d1 = second best.
        ratio = d0 / (d1 + 1e-8).
        If ratio < ratio_threshold => ambiguous => new.
        (Typically a smaller ratio means a better match, but we invert logic:
        a patch can be "new" if the ratio is extremely small or ambiguous.)  
    """

    N_src, N_tgt = distances.shape

    # Target → Source: best match
    min_vals_t, best_s_for_t = torch.min(distances, dim=0)

    # Source → Target: best match
    min_vals_s, best_t_for_s = torch.min(distances, dim=1)

    # Two-sided consistency check
    twosided_mask = torch.zeros(N_tgt, device=distances.device)
    if two_sided:
        for t in range(N_tgt):
            s = best_s_for_t[t]
            if best_t_for_s[s] != t:
                twosided_mask[t] = 1.0

    # Ratio test: ambiguous if best match is not clearly better than second best
    sorted_dists, _ = torch.sort(distances, dim=0)
    d0 = sorted_dists[0]
    d1 = sorted_dists[1]
    ratio = d0 / (d1 + 1e-8)
    ratio_threshold = torch.quantile(ratio, top_k_ratio_quantile)
    ratio_mask = (ratio < ratio_threshold).float()

    # Combine checks (currently using only two-sided result)
    newness_mask = twosided_mask

    return newness_mask


