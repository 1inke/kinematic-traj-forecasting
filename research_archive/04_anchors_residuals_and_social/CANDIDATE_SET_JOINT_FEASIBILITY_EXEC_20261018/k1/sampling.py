"""Deterministic window / state / action sampling. No GT in hashes."""
from __future__ import annotations
import hashlib
import numpy as np
from .config import N_NAT, N_KIN, N_TRAIN_WINDOWS, STATE_N_SWAPS, N_ACTIONS_PER_STATE, STATES


def _sha_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha_int(s: str) -> int:
    return int(_sha_hex(s)[:16], 16)


def select_train_windows(sample_ids: np.ndarray, fit_train_ix: np.ndarray, n=N_TRAIN_WINDOWS):
    """
    From FIT_train global indices, take n windows by SHA256('set-marginal-v1|sample_id') ascending.
    Same IDs across seeds. Returns global indices (sorted by hash order, not by id).
    """
    ids = sample_ids[fit_train_ix]
    keys = np.array([_sha_hex(f"set-marginal-v1|{int(sid)}") for sid in ids])
    order = np.argsort(keys, kind="mergesort")
    take = order[: min(n, len(order))]
    chosen = fit_train_ix[take]
    return chosen, dict(
        n_requested=n,
        n_available=int(len(fit_train_ix)),
        n_taken=int(len(chosen)),
        id_sha_first=_sha_hex(str(int(sample_ids[chosen[0]]))) if len(chosen) else None,
        selection_tag="set-marginal-v1",
    )


def _pair_key(tag: str, sid: int, nat_slot: int, kin_slot: int) -> str:
    return _sha_hex(f"{tag}|{sid}|{nat_slot}|{kin_slot}")


def build_state_selected(sid: int, n_swaps: int, tag: str):
    """
    Build selected 20 indices from native by applying n_swaps native->KIN replacements.
    Matching: sort all (nat,kin) by hash, greedily take unused.
    Returns list of 20 pool indices (order: remaining native slots kept in place? 
    Design: keep physical indices; start native 0..19, replace.
    We store as a set of 20 unique pool indices; deployment uses list with rem replaced by add.
    For state construction, start [0..19], for each accepted pair replace nat with kin.
    """
    selected = list(range(N_NAT))
    if n_swaps <= 0:
        return selected, []
    pairs = []
    for nat in range(N_NAT):
        for kin in range(N_NAT, N_NAT + N_KIN):
            pairs.append((_pair_key(tag, sid, nat, kin), nat, kin))
    pairs.sort(key=lambda x: (x[0], x[1], x[2]))
    used_n, used_k = set(), set()
    swaps = []
    for _, nat, kin in pairs:
        if nat in used_n or kin in used_k:
            continue
        # nat must still be in selected
        if nat not in selected:
            continue
        selected = [kin if x == nat else x for x in selected]
        used_n.add(nat)
        used_k.add(kin)
        swaps.append((nat, kin))
        if len(swaps) >= n_swaps:
            break
    assert len(selected) == N_NAT and len(set(selected)) == N_NAT
    return selected, swaps


def sample_action_pairs(sid: int, state_name: str, selected):
    """
    From (native still in S) x (KIN not in S), take 32 by SHA256('pair|id|state|r|l') ascending.
    """
    rem_cands = [j for j in selected if j < N_NAT]
    in_set = set(selected)
    add_cands = [j for j in range(N_NAT, N_NAT + N_KIN) if j not in in_set]
    scored = []
    for r in rem_cands:
        for l in add_cands:
            scored.append((_sha_hex(f"pair|{sid}|{state_name}|{r}|{l}"), r, l))
    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    take = scored[:N_ACTIONS_PER_STATE]
    return [(r, l) for _, r, l in take]


def enumerate_all_legal_actions(selected):
    rem_cands = [j for j in selected if j < N_NAT]
    in_set = set(selected)
    add_cands = [j for j in range(N_NAT, N_NAT + N_KIN) if j not in in_set]
    return [(r, a) for r in rem_cands for a in add_cands]


def build_all_states_for_window(sid: int):
    out = {}
    for st in STATES:
        n = STATE_N_SWAPS[st]
        tag = "state4" if st == "S4" else ("state12" if st == "S12" else "state0")
        # S0 uses no swaps; tag unused
        sel, swaps = build_state_selected(sid, n, tag)
        pairs = sample_action_pairs(sid, st, sel)
        out[st] = dict(selected=sel, swaps=swaps, pairs=pairs)
    return out
