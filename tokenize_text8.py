import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
raw_path = ROOT / "data" / "text8" / "text8"
out_path = ROOT / "data" / "text8_tokenized.json"

text = raw_path.read_text(encoding="utf-8").lower()
# keep letters and spaces only
text = re.sub(r"[^a-z\s]+", " ", text)
tokens = text.split()

chunk_size = 50
sentences = []
for i in range(0, len(tokens), chunk_size):
    chunk = tokens[i:i+chunk_size]
    if len(chunk) >= 5:
        sentences.append(chunk)

out_path.write_text(json.dumps(sentences), encoding="utf-8")
print(f"wrote {len(sentences)} chunks to {out_path}")