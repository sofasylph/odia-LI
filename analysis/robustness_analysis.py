"""
Robustness analysis for Group 4 (IndicLID on Bhasha-Abhijnaanam + native Odia).

Regenerates every number in results/robustness_analysis.json:
  1. independent verification of Track B / Track C metrics from the prediction files
  2. translationese gap with bootstrap CI, and a length-matched gap
  3. Odia vocabulary coverage of the fine-tuned model's training data
  4. emoji ablation (diagnostic only)
  5. the PUBLISHED IndicLID system scored on the native Odia set
  6. false-"Odia" rates on full-length and YouTube-length non-Odia text, balanced accuracy

Nothing here selects or tunes anything; the reported Track C number is fixed by the notebook.

Usage (from repo root):
    python analysis/robustness_analysis.py
Expects (see data/README.md and models/README.md):
    data/bhasha-abhijnaanam.json
    data/04_romanized_lid_group_4.jsonl          (native set - not distributed)
    models/IndicLID-FTR.bin                      (published FastText)
    models/indiclid-bert.pt                      (published BERT)
    models/indiclid_ftr_finetuned.bin            (our fine-tuned FastText)
    results/predictions_benchmark_test.jsonl
"""
import argparse, collections, json, random, re, types
import numpy as np

# --- fasttext 0.9.x vs numpy 2.x: scoped shim, never patch numpy globally ---------------
import fasttext.FastText as FT


class _NumpyShim(types.ModuleType):
    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def array(obj, *a, **k):
        if k.get("copy") is False:
            k["copy"] = None
        return np.array(obj, *a, **k)


FT.np = _NumpyShim("numpy_shim")
import fasttext  # noqa: E402

L2C = {'Assamese': 'asm', 'Bangla': 'ben', 'Bodo': 'brx', 'Gujarati': 'guj', 'Hindi': 'hin',
       'Kannada': 'kan', 'Kashmiri': 'kas', 'Konkani': 'kok', 'Maithili': 'mai',
       'Malayalam': 'mal', 'Marathi': 'mar', 'Manipuri': 'mni', 'Nepali': 'nep',
       'Oriya': 'ori', 'Punjabi': 'pan', 'Sanskrit': 'san', 'Sindhi': 'snd',
       'Tamil': 'tam', 'Telugu': 'tel', 'Urdu': 'urd'}
BERT_REV = {0: 'asm', 1: 'ben', 2: 'brx', 3: 'guj', 4: 'hin', 5: 'kan', 6: 'kas', 7: 'kok',
            8: 'mai', 9: 'mal', 10: 'mni', 11: 'mar', 12: 'nep', 13: 'ori', 14: 'pan',
            15: 'san', 16: 'snd', 17: 'tam', 18: 'tel', 19: 'urd', 20: 'eng', 21: 'other'}
WS = re.compile(r"\s+")
EMO = re.compile(r'[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200d]')
SPECIAL = re.compile(r'[@_!#$%^&*()<>?/\\|}{~:]')
EN = re.compile(r'[a-zA-Z0-9]')


def normalize(t):                       # identical to the notebook's Track B preprocessing
    return WS.sub(" ", t.replace("\n", " ")).strip().lower()


def roman_share(s):                     # identical to the notebook / official IndicLID gate
    t = len(s) - len(SPECIAL.findall(s)) - len(re.findall(r'\s', s))
    return 0 if t <= 0 else len(EN.findall(s)) / t


def tokens(s):
    return [w.strip(".,!?;:'\"()") for w in EMO.sub(" ", s).split() if w.strip(".,!?;:'\"()")]


def r2(x):
    return None if x is None else round(float(x), 4)


def load_benchmark(path):
    d = json.load(open(path))["data"]
    by = collections.defaultdict(list)
    for r in d:
        if r.get("language") in L2C and r.get("romanized sentence", "").strip():
            by[r["language"]].append({"id": r["unique_identifier"],
                                      "raw": r["romanized sentence"].replace("\n", " "),
                                      "text": normalize(r["romanized sentence"]),
                                      "label": L2C[r["language"]]})
    rng = random.Random(42)              # same seed and per-language 80/10/10 as the notebook
    sp = {"train": [], "valid": [], "test": []}
    for lang, items in sorted(by.items()):
        items = sorted(items, key=lambda x: x["id"]); rng.shuffle(items)
        n = len(items); a = int(n * .8); b = int(n * .9) - a
        sp["train"] += items[:a]; sp["valid"] += items[a:a + b]; sp["test"] += items[a + b:]
    return sp


def ft_predict(model, texts):
    lab, sc = model.predict(list(texts))
    return ([l[0].replace("__label__", "").replace("_Latn", "") for l in lab],
            np.array([s[0] for s in sc]))


def main(a):
    out = {}
    sp = load_benchmark(a.benchmark)
    gold_all = [json.loads(l) for l in open(a.native) if l.strip()]
    C = [json.loads(l) for l in open(a.native_preds)]
    # the distributed predictions file carries no comment text; join it from the native set
    by_id = {r["id"]: r for r in gold_all}
    assert len(by_id) == len(gold_all), "native set ids are not unique"
    for r in C:
        g = by_id[r["id"]]
        r.setdefault("text", g["text"]); r.setdefault("variant", g["variant"])
    B = [json.loads(l) for l in open(a.bench_preds)]

    # ------------------------------------------------ 1. verification
    yb = [r["gold"] for r in B]; pb = [r["pred"] for r in B]
    tp = sum(g == "ori" and p == "ori" for g, p in zip(yb, pb))
    fp = sum(g != "ori" and p == "ori" for g, p in zip(yb, pb))
    fn = sum(g == "ori" and p != "ori" for g, p in zip(yb, pb))
    from sklearn.metrics import f1_score
    ok = np.array([r["pred"] == "ori" for r in C])

    def slice_acc(idx):
        return {"n": len(idx), "correct": int(ok[idx].sum()), "accuracy": r2(ok[idx].mean() * 100)}
    lens_c = [len(r["text"].split()) for r in C]
    out["verification"] = {
        "track_B": {"n": len(B), "accuracy": r2(np.mean([g == p for g, p in zip(yb, pb)]) * 100),
                    "macro_f1": r2(f1_score(yb, pb, average="macro", zero_division=0) * 100),
                    "odia_tp": tp, "odia_fp": fp, "odia_fn": fn},
        "track_C": {"n": len(C), "correct": int(ok.sum()), "accuracy_eq_odia_recall": r2(ok.mean() * 100),
                    "sent_to_bert_pct": r2(np.mean([r["routed"] == "BERT" for r in C]) * 100),
                    "by_variant": {v: slice_acc([i for i, r in enumerate(C) if r["variant"] == v])
                                   for v in ("roman", "mixed")},
                    "by_length": {"<5": slice_acc([i for i, n in enumerate(lens_c) if n < 5]),
                                  "5-9": slice_acc([i for i, n in enumerate(lens_c) if 5 <= n < 10]),
                                  ">=10": slice_acc([i for i, n in enumerate(lens_c) if n >= 10])},
                    "by_route": {rt: slice_acc([i for i, r in enumerate(C) if r["routed"] == rt])
                                 for rt in ("FTR", "BERT")}}}

    # ------------------------------------------------ 2. gaps
    b = np.array([r["pred"] == "ori" for r in B if r["gold"] == "ori"])
    rng = np.random.default_rng(0); K = 5000
    gb = np.array([b[rng.integers(0, len(b), len(b))].mean() for _ in range(K)]) * 100
    gc = np.array([ok[rng.integers(0, len(ok), len(ok))].mean() for _ in range(K)]) * 100
    gap = gb - gc
    odia_text = {r["id"]: r["text"] for v in sp.values() for r in v if r["label"] == "ori"}
    Bo = [(len(odia_text[r["id"]].split()), r["pred"] == "ori") for r in B if r["gold"] == "ori"]
    Co = list(zip(lens_c, ok))
    b10 = [o for n, o in Bo if n >= 10]; c10 = [o for n, o in Co if n >= 10]
    out["translationese_gap"] = {
        "odia_recall_like_for_like": {"benchmark": r2(b.mean() * 100), "native": r2(ok.mean() * 100),
                                      "gap": r2((b.mean() - ok.mean()) * 100),
                                      "ci95": [r2(np.percentile(gap, 2.5)), r2(np.percentile(gap, 97.5))],
                                      "resamples_with_gap_le_0": int((gap <= 0).sum()), "resamples": K},
        "length_matched_ge10_words": {"benchmark_n": len(b10), "native_n": len(c10),
                                      "benchmark": r2(np.mean(b10) * 100), "native": r2(np.mean(c10) * 100),
                                      "gap": r2((np.mean(b10) - np.mean(c10)) * 100)}}

    # ------------------------------------------------ 3. vocabulary coverage
    ori_v = collections.Counter(w for r in sp["train"] if r["label"] == "ori" for w in tokens(r["text"]))
    oth_v = collections.Counter(w for r in sp["train"] if r["label"] != "ori" for w in tokens(r["text"]))

    def cover(texts):
        toks = [w for t in texts for w in tokens(t)]
        return {"tokens": len(toks),
                "seen_in_odia_training_pct": r2(sum(w in ori_v for w in toks) / len(toks) * 100),
                "seen_only_in_other_languages_pct":
                    r2(sum((w in oth_v) and (w not in ori_v) for w in toks) / len(toks) * 100)}
    nat = collections.Counter(w for r in C for w in tokens(normalize(r["text"])))
    lang_of = collections.defaultdict(collections.Counter)
    for r in sp["train"]:
        if r["label"] != "ori":
            for w in tokens(r["text"]): lang_of[w][r["label"]] += 1
    examples = []
    for w, c in nat.most_common(400):
        if w not in ori_v and w in lang_of and len(w) > 2:
            examples.append({"word": w, "count_native": c, "seen_in": dict(lang_of[w].most_common(2))})
        if len(examples) == 10: break
    out["vocabulary"] = {
        "odia_training_sentences": sum(r["label"] == "ori" for r in sp["train"]),
        "odia_training_distinct_words": len(ori_v),
        "other_training_sentences": sum(r["label"] != "ori" for r in sp["train"]),
        "other_training_distinct_words": len(oth_v),
        "benchmark_odia_test": cover([r["text"] for r in sp["test"] if r["label"] == "ori"]),
        "native_odia": cover([normalize(r["text"]) for r in C]),
        "common_native_words_unseen_in_odia_training": examples}

    # ------------------------------------------------ models
    ourF = fasttext.load_model(a.our_ftr)
    pubF = fasttext.load_model(a.pub_ftr)

    # ------------------------------------------------ 4. emoji ablation
    Xn = [normalize(r["text"]) for r in C]
    Xe = [WS.sub(" ", EMO.sub(" ", t)).strip() for t in Xn]
    acc = lambda m, T: r2(np.mean([p == "ori" for p in ft_predict(m, T)[0]]) * 100)
    out["emoji_ablation_diagnostic_only"] = {
        "rows_with_emoji": sum(bool(EMO.search(r["text"])) for r in C),
        "our_fasttext_as_preprocessed": acc(ourF, Xn),
        "our_fasttext_emoji_stripped": acc(ourF, Xe)}

    # ------------------------------------------------ BERT (published)
    import torch
    from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
    mp = torch.load(a.pub_bert, map_location="cpu", weights_only=False)
    sd = mp.state_dict(); del mp
    cfg = AutoConfig.from_pretrained("ai4bharat/IndicBERTv2-MLM-only", num_labels=22)
    bert = AutoModelForSequenceClassification.from_config(cfg)
    miss, _ = bert.load_state_dict(sd, strict=False, assign=True); assert not miss; del sd
    bert.eval(); tok = AutoTokenizer.from_pretrained("ai4bharat/IndicBERTv2-MLM-only")
    cache = {}

    @torch.no_grad()
    def bert_predict(T):
        need = [t for t in dict.fromkeys(T) if t not in cache]
        for i in range(0, len(need), 16):
            e = tok(need[i:i + 16], return_tensors="pt", padding=True, truncation=True, max_length=256)
            for t, k in zip(need[i:i + 16], torch.argmax(
                    bert(input_ids=e["input_ids"], attention_mask=e["attention_mask"]).logits, 1)):
                cache[t] = BERT_REV[k.item()]
        return [cache[t] for t in T]

    def pub_ensemble(T, thr):
        p, c = ft_predict(pubF, T)
        sh = np.array([roman_share(t) for t in T])
        esc = (c <= thr) & (sh > 0.5)
        idx = list(np.where(esc)[0]); bp = dict(zip(idx, bert_predict([T[i] for i in idx])))
        return [bp[i] if esc[i] else p[i] for i in range(len(T))], float(esc.mean() * 100)

    # ------------------------------------------------ 5. published system on native
    pos = [r["text"].replace("\n", " ") for r in gold_all if r["quality"] == "valid_odia"]
    rej = [r["text"].replace("\n", " ") for r in gold_all if r["quality"] == "reject"]
    rate = lambda P: r2(np.mean([p == "ori" for p in P]) * 100)
    e06, s06 = pub_ensemble(pos, 0.6); e95, s95 = pub_ensemble(pos, 0.95)
    out["published_system_on_native"] = {
        "n": len(pos), "note": "raw text, as in Track A",
        "fasttext_alone": rate(ft_predict(pubF, pos)[0]),
        "bert_alone": rate(bert_predict(pos)),
        "ensemble_0.6": {"odia_recall": rate(e06), "sent_to_bert_pct": r2(s06)},
        "ensemble_0.95": {"odia_recall": rate(e95), "sent_to_bert_pct": r2(s95)}}

    # ------------------------------------------------ 6. false alarms + balanced accuracy
    test = []
    by = collections.defaultdict(list)
    for r in sp["test"]: by[r["label"]].append(r)
    d = json.load(open(a.benchmark))["data"]       # rebuild the held-out 10% in the same order
    by2 = collections.defaultdict(list)            # used by the original analysis run
    for r in d:
        if r.get("language") in L2C and r.get("romanized sentence", "").strip():
            by2[r["language"]].append({"id": r["unique_identifier"],
                                       "raw": r["romanized sentence"].replace("\n", " "),
                                       "label": L2C[r["language"]]})
    rg = random.Random(42)
    for lang, items in sorted(by2.items()):
        items = sorted(items, key=lambda x: x["id"]); rg.shuffle(items)
        test += items[int(len(items) * .9):]
    neg_full = [r["raw"] for r in test if r["label"] != "ori"]
    lens_pos = [len(t.split()) for t in pos]
    rs7 = random.Random(7)
    neg_full = rs7.sample(neg_full, 2000)
    neg_short = [" ".join(t.split()[:rs7.choice(lens_pos)]) for t in neg_full]

    res = {}
    for name, fn_ in (("published_fasttext", lambda T: ft_predict(pubF, T)[0]),
                      ("published_ensemble_0.6", lambda T: pub_ensemble(T, 0.6)[0]),
                      ("our_fasttext", lambda T: ft_predict(ourF, [normalize(t) for t in T])[0])):
        tpr = rate(fn_(pos)); ff = rate(fn_(neg_full)); fs = rate(fn_(neg_short)); rj = rate(fn_(rej))
        res[name] = {"catches_odia_native": tpr, "false_odia_full_length": ff,
                     "false_odia_short_text": fs, "balanced_accuracy_short": r2((tpr + 100 - fs) / 2),
                     "calls_rejects_odia": rj}
    our_ba = res["our_fasttext"]["balanced_accuracy_short"]
    for name in ("published_fasttext", "published_ensemble_0.6"):
        res[name]["false_odia_rate_needed_to_match_ours"] = r2(
            res[name]["catches_odia_native"] + 100 - 2 * our_ba)
    out["false_alarm_analysis"] = {
        "negatives": {"source": "held-out 10% non-Odia benchmark sentences, never trained on",
                      "n": len(neg_full),
                      "median_words_full": float(np.median([len(t.split()) for t in neg_full])),
                      "median_words_short": float(np.median([len(t.split()) for t in neg_short])),
                      "median_words_native": float(np.median(lens_pos))},
        "rejects": {"n": len(rej), "variant": dict(collections.Counter(
            r["variant"] for r in gold_all if r["quality"] == "reject")),
            "note": "mostly code-mixed Odia, so NOT usable as negatives"},
        "models": res,
        "our_ensemble_note": "not scorable on negatives: fine-tuned BERT weights not included"}

    json.dump(out, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="data/bhasha-abhijnaanam.json")
    p.add_argument("--native", default="data/04_romanized_lid_group_4.jsonl")
    p.add_argument("--native_preds", default="results/predictions_native_odia.jsonl")
    p.add_argument("--bench_preds", default="results/predictions_benchmark_test.jsonl")
    p.add_argument("--our_ftr", default="models/indiclid_ftr_finetuned.bin")
    p.add_argument("--pub_ftr", default="models/IndicLID-FTR.bin")
    p.add_argument("--pub_bert", default="models/indiclid-bert.pt")
    p.add_argument("--out", default="results/robustness_analysis.json")
    main(p.parse_args())
