```text
TXT Files
    │
    ▼
Paragraph Extraction
    │
    ▼
Sentence-aware Split
(≤450 tokens)
    │
    ▼
Chunk Objects
    │
    ▼
Check Cache
    │
 ┌──┴───────────────┐
 │                  │
Cache Exists?     No Cache
 │                  │
 ▼                  ▼
Load JSON      Call Advanced LLM
 │                  │
 └───────┬──────────┘
         ▼
Topic + 5 Diverse Questions
         │
         ▼
Generate 5 JSONL Samples
(one per question)
         │
         ▼
train.jsonl
```

## Adapter training

```text
train.jsonl
    |
    v
Validate chat records
    |
    v
Chunk-grouped train/validation split
    |
    v
Load Qwen in 4-bit mode
    |
    v
QLoRA supervised fine-tuning
    |-----------------------> models/checkpoints/
    v
Save adapter + tokenizer
    |
    v
models/adapters/
    |
    v
app.py loads base model + adapter
```

Validate the dataset and configuration without loading the model:

```powershell
python scripts/fine_tune.py --validate-only
```

Start a new training run:

```powershell
python scripts/fine_tune.py
```

Resume the newest checkpoint:

```powershell
python scripts/fine_tune.py --resume
```
