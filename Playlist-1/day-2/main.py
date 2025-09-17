import json, os, re
from collections import Counter, defaultdict
from rapidfuzz import fuzz, process

FAQ_PATH = "faq.jsonl"
SHORTLIST_K = 8
WEIGHT_PARTIAL = 0.45
WEIGHT_TOKEN_SET = 0.35
WEIGHT_TOKEN_SORT = 0.20
WEIGHT_CHAR = 0.25   
MIN_ACCEPT_SCORE = 0.58

def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def tokens(s: str):
    return normalize(s).split()

def char_trigrams(s: str):
    s = normalize(s).replace(" ", "_")
    if len(s) < 3:
        return {s}
    return {s[i:i+3] for i in range(len(s)-2)}

def jaccard(a:set, b:set):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union

def load_faq(path=FAQ_PATH):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                out.append(json.loads(line))
            except:
                continue
    return out

faq = load_faq()
questions = [q["question"] for q in faq]
norm_questions = [normalize(q) for q in questions]

inverted = defaultdict(set)
for i, q in enumerate(norm_questions):
    for t in set(q.split()):
        inverted[t].add(i)

def shortlist(query, k=SHORTLIST_K):
    toks = set(tokens(query))
    counts = Counter()
    for t in toks:
        for idx in inverted.get(t, ()):
            counts[idx] += 1
    if counts:
        return [idx for idx, _ in counts.most_common(k)]
    
    best = process.extract(query, questions, scorer=fuzz.partial_ratio, limit=k)
    return [questions.index(b[0]) for b in best]

def fuzzy_ensemble_scores(query, cand_qs):
    out = []
    for cand in cand_qs:
        p = fuzz.partial_ratio(query, cand) / 100.0
        ts = fuzz.token_set_ratio(query, cand) / 100.0
        tsort = fuzz.token_sort_ratio(query, cand) / 100.0
        score = WEIGHT_PARTIAL*p + WEIGHT_TOKEN_SET*ts + WEIGHT_TOKEN_SORT*tsort
        out.append(score)
    return out

def find_answer(query, debug=False):
    if not faq:
        return "No knowledge base loaded.", {"strategy":"none"}

    nq = normalize(query)

    for i, nq_q in enumerate(norm_questions):
        if nq in nq_q or nq_q in nq:
            return faq[i]["answer"], {"strategy":"substring","idx":i}

    cands = shortlist(query)
    cand_qs = [questions[i] for i in cands]
    fuzzy_scores = fuzzy_ensemble_scores(query, cand_qs)
    char_scores = []
    q_tris = char_trigrams(query)
    for i in cands:
        char_scores.append(jaccard(q_tris, char_trigrams(questions[i])))

    combined = []
    for j, idx in enumerate(cands):
        fuzzy_part = fuzzy_scores[j]
        comb = WEIGHT_CHAR * char_scores[j] + (1 - WEIGHT_CHAR) * fuzzy_part
        combined.append((idx, comb, fuzzy_part, char_scores[j]))

    combined.sort(key=lambda x: x[1], reverse=True)
    best_idx, best_score, best_fuzzy, best_char = combined[0]

    if len(tokens(query)) <= 3:
        best_score += 0.04 * best_fuzzy

    if debug:
        print("[debug] candidates:")
        for c in combined[:6]:
            print(f"  idx={c[0]} score={c[1]:.3f} fuzzy={c[2]:.3f} char={c[3]:.3f} q={questions[c[0]]!r}")

    if best_score >= MIN_ACCEPT_SCORE:
        return faq[best_idx]["answer"], {"strategy":"enhanced_fuzzy","idx":best_idx,"score":best_score}
    return "Sorry, I don't know that yet.", {"strategy":"fallback","best_score":best_score}

if __name__ == "__main__":
    print("Light enhanced fuzzy chatbot")
    while True:
        u = input("You: ").strip()
        if not u: continue
        if u.lower() in ("quit","exit"):
            break
        ans, meta = find_answer(u, debug=True)
        if meta.get("strategy") == "enhanced_fuzzy":
            print(f"Bot [score={meta['score']:.3f}]: {ans}")
        else:
            print("Bot:", ans)
