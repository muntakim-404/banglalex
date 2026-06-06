import json, os, time, random
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY",""))
PATH   = "data/annotated/cases_augmented.json"
TARGET = {"land": 133, "contract": 133, "service": 134}

with open(PATH, "r", encoding="utf-8") as f:
    all_cases = json.load(f)

real = [c for c in all_cases if not c.get("is_augmented")]
aug  = [c for c in all_cases if c.get("is_augmented")]

for domain in ["land","contract","service"]:
    r = [c for c in real if c["domain"]==domain]
    a = [c for c in aug  if c["domain"]==domain]
    needed = TARGET[domain] - len(r) - len(a)
    if needed <= 0:
        print(f"{domain}: ✓ already at target ({len(r)+len(a)})")
        continue

    max_copy = max((int(c["citation"].split("-")[0].replace("AUG",""))
                    for c in a if c["citation"].startswith("AUG")), default=0)
    copy_num = max_copy + 1
    subset   = random.sample(r, min(needed, len(r)))

    print(f"{domain}: generating {needed} more (copy {copy_num})")
    for i, case in enumerate(subset):
        print(f"  [{i+1}/{needed}] {case['citation'][:45]}", end="\r")
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role":"user","content":
                    f"Rewrite this {domain} law case summary in different words. "
                    "Keep all facts, parties, statutes and outcome identical. "
                    "Output only the rewritten summary.\n\n"+case["facts_summary"]}],
                temperature=0.8, max_tokens=500)
            c2 = case.copy()
            c2["citation"]        = f"AUG{copy_num}-{case['citation']}"
            c2["facts_summary"]   = resp.choices[0].message.content.strip()
            c2["is_augmented"]    = True
            c2["source_citation"] = case["citation"]
            aug.append(c2)
            time.sleep(0.8)
        except Exception as e:
            print(f"\n  error: {e}")

all_cases = real + aug
with open(PATH,"w",encoding="utf-8") as f:
    json.dump(all_cases, f, ensure_ascii=False, indent=2)

print(f"\nTotal: {len(all_cases)}")
for d in ["land","contract","service"]:
    r=len([c for c in real if c["domain"]==d])
    a=len([c for c in aug  if c["domain"]==d])
    print(f"  {d}: {r} real + {a} aug = {r+a}")