# Models

| File | In repo? | Size | Source |
|---|---|---|---|
| `indiclid_ftr_finetuned.bin` | **Yes** | 75.5 MB | Ours — fine-tuned FastText, Track B/C stage 1 |
| `indiclid_bert_finetuned.pt` | No | 1.1 GB | Ours — fine-tuned IndicBERT, Track B/C stage 2 |
| `IndicLID-FTR.bin` | No | 270 MB | Published — https://huggingface.co/ai4bharat/IndicLID-FTR (`model_baseline_roman.bin`) |
| `indiclid-bert.pt` | No | 1.1 GB | Published — https://huggingface.co/ai4bharat/IndicLID-BERT (`basline_nn_simple.pt`) |

**Our fine-tuned FastText** — sha256 `dae480af16837f877cb8a43e6fe9ae91848f53aed53b40c2247ced9fc4276bfd`.
Verified to reproduce 95.81 validation macro F1 and 95.78% test accuracy on the notebook's split.
At 75.5 MB it is under GitHub's 100 MB file limit, though GitHub warns above 50 MB; tracking it
with Git LFS (`git lfs track "*.bin"`) is cleaner.

**Our fine-tuned IndicBERT** exceeds GitHub's file limit. Attach it to a GitHub Release
(up to 2 GB per file) or upload it to the Hugging Face Hub, and link it here.

**Published models** are downloaded by the notebook automatically. For the analysis script,
place them in this folder under the names above. Expected sha256:
- `IndicLID-FTR.bin`: `b691f6c44ad1ba04e18d1d8c649ecd3f7f091075600c9d78f3ce5d6c930fa7bf`
- `indiclid-bert.pt`: `38462faf42b8b0a5ef66a548abffcf6b7e91f5f5576b54d39a6de96da4a4ce1c`

The published BERT checkpoint is a pickled 2023 transformers object that does not run on
current transformers. Both the notebook and the analysis script load it once, extract
`state_dict()`, and rebuild the model from `AutoConfig`.
