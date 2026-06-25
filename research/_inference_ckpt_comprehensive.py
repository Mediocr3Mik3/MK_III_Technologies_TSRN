"""Comprehensive inference test: 15 prompts across 3 categories.

Category A — In-domain (tropical geometry / math / CS): 5 prompts
Category B — Out-of-domain (cooking / sports / poetry / etc.): 5 prompts
Category C — Long-form past context window (stress-test memory): 5 prompts

For Category C we generate up to 2x training context (512 tokens) to stress
KleeneSSM + KleeneAttention long-range recall.  We prepend a "memory" setup
and ask the model to recall it after a long intervening sequence.
"""
import sys, os, torch

_RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

from tsrn_gist import TSRNGist
from model_config import ModelConfig
from transformers import GPT2TokenizerFast
import dataclasses

CKPT_PATH = r"D:\ml\tsrn_data\checkpoints\20260619_v2_kleene_directml_osc_d256_ctx256_step0120000.pt"
TOK_PATH = r"D:\ml\tsrn_data\tokenizer\gpt2_tsrn_base"
CTX_LEN = 256

PROMPTS = {
    # ---- Category A: In-domain (tropical geometry / math / CS / Kleene algebra)
    "A1": "The tropical semiring, defined over the real numbers with max as addition and ordinary addition as multiplication,",
    "A2": "In tropical geometry, a tropical polynomial is a piecewise-linear function formed by taking the maximum of affine functions. This implies that",
    "A3": "The Kleene star operation in a weighted automaton computes the shortest path between all pairs of states by iteratively applying",
    "A4": "When we apply Maslov dequantization to the softmax function, the limit as the temperature parameter h approaches zero yields",
    "A5": "The Bellman optimality principle states that an optimal policy has the property that whatever the initial state and initial decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision. In tropical terms,",

    # ---- Category B: Out-of-domain (cooking / sports / poetry / weather / travel)
    "B1": "To prepare a classic French coq au vin, first brown the chicken pieces in butter, then add onions, mushrooms, and a bottle of good red Burgundy. Simmer gently for two hours until",
    "B2": "In the final seconds of overtime, the quarterback dropped back, scanned the field, and launched a deep pass toward the end zone. The crowd held its breath as",
    "B3": "The autumn wind whispered through the maple trees, scattering crimson leaves across the cobblestone path. An old woman sat on the park bench, remembering the summer of 1967 when",
    "B4": "Tourists visiting Kyoto in April should arrive early at Kiyomizu-dera to avoid the cherry blossom crowds. The temple's wooden stage offers a panoramic view of",
    "B5": "Modern sous-vide cooking relies on precise temperature control to denature proteins without overcooking. For a medium-rare steak, set the water bath to 54 degrees Celsius and cook for",

    # ---- Category C: Long-form memory stress-test (recall after long generation)
    "C1": "STORY_SETUP: Dr. Elara Voss discovered a tropical number system in which every equation had a unique solution. DECADE_LATER: Ten years passed. Dr. Voss stood before the academy and declared that her greatest achievement was",
    "C2": "CHARACTER_MEMO: Agent K-7's secret codeword is TROPICALIZER. After a lengthy chase across three continents, the villain finally cornered the agent and demanded the codeword. Agent K-7 whispered",
    "C3": "MATH_PUZZLE: Solve for x where max(x+3, 7) = 9. EXPLANATION_BEGIN: To solve this tropical equation, we first consider the two cases. In the first case, x+3 >= 7, so max(x+3, 7) = x+3. Setting x+3 = 9 gives x = 6, which satisfies x >= 4. In the second case, x+3 < 7, so max(x+3, 7) = 7, which cannot equal 9. Therefore, the unique solution is x = 6. NOW_VERIFY: Re-checking, if x = 6 then x+3 = 9 and max(9, 7) = 9. Correct. FINAL_ANSWER:",
    "C4": "FACT_BANK: The Maslov temperature h controls the interpolation between classical softmax (h=1) and hard max-plus (h->0). When h is very small, the attention scores approximate a winner-take-all routing. SCENARIO: During inference, a developer sets h = 0.05 and observes that the attention pattern becomes",
    "C5": "PLOT: In 2147, the colony ship Tropical Horizon used a Kleene-star navigation matrix to find shortest paths through the asteroid belt. After seventy years of drift, the ship's AI rebooted and calculated the safest route by applying the",
}

CATEGORY_INFO = {
    "A": "In-domain (tropical geometry / math / CS / Kleene algebra)",
    "B": "Out-of-domain (cooking / sports / poetry / travel / science)",
    "C": "Long-form memory stress-test (recall after long generation)",
}


def load_model():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    valid_fields = {f.name for f in dataclasses.fields(ModelConfig)}
    model_cfg = ModelConfig(**{k: v for k, v in cfg.items() if k in valid_fields})
    if not hasattr(model_cfg, "reservoir_kind"):
        model_cfg.reservoir_kind = "echo"
    if not hasattr(model_cfg, "use_linear_attention"):
        model_cfg.use_linear_attention = False

    model = TSRNGist(
        vocab=model_cfg.vocab_size,
        d_model=model_cfg.d_model,
        context_len=model_cfg.context_len,
        gradient_checkpoint=False,
        n_blocks=model_cfg.n_blocks,
        top_k=model_cfg.top_k,
        n_heads=model_cfg.n_heads,
        mem_depth=getattr(model_cfg, "padic_depth", 5),
        max_gists=getattr(model_cfg, "max_gists", 32),
        gist_top_k=getattr(model_cfg, "gist_top_k", 4),
        dropout=getattr(model_cfg, "dropout", 0.0),
        use_hyperbolic=False,
        gist_chaining=False,
        config=model_cfg,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    try:
        import torch_directml as _dml
        device = _dml.device()
    except ImportError:
        device = torch.device("cpu")
    model = model.to(device).eval()
    return model, device


REPETITION_PENALTY = 1.3
NO_REPEAT_NGRAM = 3
TOP_P = 0.9


def _apply_repetition_penalty(logits, generated_ids, penalty):
    for token_id in set(generated_ids):
        if logits[token_id] < 0:
            logits[token_id] *= penalty
        else:
            logits[token_id] /= penalty
    return logits


def _apply_no_repeat_ngram(logits, generated_ids, n):
    if n <= 0 or len(generated_ids) < n:
        return logits
    prefix = tuple(generated_ids[-(n - 1):])
    banned = {
        generated_ids[i + n - 1]
        for i in range(len(generated_ids) - n + 1)
        if tuple(generated_ids[i : i + n - 1]) == prefix
    }
    for t in banned:
        logits[t] = float("-inf")
    return logits


def _top_p_filter(logits, top_p):
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    cutoff = cum_probs > top_p
    if cutoff.any():
        first_idx = cutoff.nonzero(as_tuple=True)[0][0].item()
        sorted_logits[first_idx + 1 :] = float("-inf")
    new_logits = torch.full_like(logits, float("-inf"))
    new_logits[sorted_idx] = sorted_logits
    return new_logits


def generate(model, tokenizer, device, prompt, max_new, temp=0.8):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([ids], device=device)
    with torch.no_grad():
        for _ in range(max_new):
            if input_ids.size(1) > CTX_LEN:
                input_ids = input_ids[:, -CTX_LEN:]
            logits, _ = model(input_ids)
            next_logits = logits[0, -1, :].clone() / temp
            next_logits = _apply_repetition_penalty(next_logits, ids, REPETITION_PENALTY)
            next_logits = _apply_no_repeat_ngram(next_logits, ids, NO_REPEAT_NGRAM)
            next_logits = _top_p_filter(next_logits, TOP_P)
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            ids.append(next_id)
            input_ids = torch.tensor([ids], device=device)
            if next_id == tokenizer.eos_token_id:
                break
    return tokenizer.decode(ids, skip_special_tokens=True)


def main():
    tokenizer = GPT2TokenizerFast.from_pretrained(TOK_PATH)
    print(f"Tokenizer vocab: {len(tokenizer)}")
    print("Loading checkpoint...")
    model, device = load_model()
    print(f"Model on device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    for cat in ["A", "B", "C"]:
        print("=" * 78)
        print(f"  CATEGORY {cat}: {CATEGORY_INFO[cat]}")
        print("=" * 78)
        for key in sorted(PROMPTS.keys()):
            if not key.startswith(cat):
                continue
            prompt = PROMPTS[key]
            # Category C gets longer generation to stress memory
            max_new = 512 if cat == "C" else 128
            text = generate(model, tokenizer, device, prompt, max_new=max_new, temp=0.8)
            generated = text[len(prompt):]
            n_new = len(tokenizer.encode(generated, add_special_tokens=False))
            print(f"\n--- {key} ({n_new} new tokens) ---")
            print(f"PROMPT: {prompt}")
            safe = generated.strip()
            safe = safe.encode("ascii", errors="replace").decode("ascii")
            print(f"OUTPUT: {safe}")
        print("\n")

    print("=" * 78)
    print("Done.")


if __name__ == "__main__":
    main()
