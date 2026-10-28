# Data

Nothing in this folder is committed.

**Bhasha-Abhijnaanam** (public). Downloaded automatically by the notebook. For the analysis
script, place `bhasha-abhijnaanam.json` here:

```bash
wget https://huggingface.co/datasets/ai4bharat/Bhasha-Abhijnaanam/resolve/main/bhasha-abhijnaanam.zip
unzip bhasha-abhijnaanam.zip
```

**Native Odia set** (`04_romanized_lid_group_4.jsonl`, not distributed). 800 YouTube comments
provided for the course and annotated by the group. It is not redistributed here because it
contains user-generated text. `results/predictions_native_odia.jsonl` keeps the predictions,
routing and slice information without the comment text; the analysis script joins the text
back by `id` when the native set is present locally.

Fields used: `id`, `text`, `quality` (`valid_odia` | `reject`), `variant` (`roman` | `mixed`).
Only `valid_odia` rows (756) are scored.
