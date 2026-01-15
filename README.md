## IKEA ASM Dataset (with uncertain event-log export for process mining)

This repository is a fork of the official **IKEA ASM** dataset codebase \[[project website](https://ikeaasm.github.io/)\].
It additionally contains **reproducible scripts** to derive **uncertain event logs** (CSV + XES) from the released
action-recognition model outputs, as used in our process-mining evaluation.

**The IKEA ASM Dataset**: Understanding People Assembling Furniture through Actions, Objects and Pose
-------------------------------------------------------------------------------------------------

### Uncertain event logs used in our paper (IKEA ASM test split)

In our process-mining evaluation, we construct uncertain event logs from the **IKEA ASM test split** (117 process instances).
We use the **pretrained per-frame predictions released by the original IKEA ASM authors** together with the action annotations.

This repo provides scripts to export:

- **Frame-wise data**: `frames.csv` (GT label + predicted distribution per frame)
- **Ground-truth log (certain)**: `segments_gt.csv` (merge consecutive identical GT labels)
- **Uncertain log (pred-merged)**: `segments_pred.csv` (merge consecutive identical predicted top-1 labels)
- **XES exports**: `xes_uncertain_gt.xes` (GT as `concept:name`) and `xes_uncertain_pred_merged.xes` (pred as `concept:name`)
  with the full distribution stored in the event attribute `probs_json`.

**Paper artifacts (ready to inspect).** The exact XES logs and segment CSVs used in our paper are included in this repository under:

- `paper_event_logs/ikea_asm/split=test/pred_merged/`: **uncertain pred-merged** logs (our main setting). Files are stored as
  `*.csv.gz` / `*.xes.gz` to keep the repository size manageable.
- `paper_event_logs/ikea_asm/split=test/gt_aligned/`: **GT-aligned** segment logs (mainly for reference / debugging).

To unpack the compressed paper logs:

```bash
gunzip -k paper_event_logs/ikea_asm/split=test/pred_merged/*.gz
```

### Installation

For the **test-split event logs used in our paper**, you only need:

- [Annotations - action](https://drive.google.com/file/d/1SwBNLViktSpk99jhh3sMXVGTMVr6tpju/view?usp=sharing) (`action_annotations.zip`)
- [Pretrained models - action](https://drive.google.com/file/d/1QksK_Uvty6pTYoGmBGWYYG3scvM_NX2X/view?usp=sharing) (`pt_models_action_recognition.zip`)

Other parts of the dataset (raw videos, pose/segmentation annotations, etc.) are only required for training/inference and the original benchmarks.

For dependencies see `requirements.txt`.

### Event-log construction and export

#### Key idea (frames → segments → XES)

For each video (one process instance), we start from **frame-wise** data at 25 fps. Each frame has:

- a **ground-truth** activity label (33 labels, including `NA`)
- a **model prediction** as a probability distribution over the same label set

We then build an event log by aggregating frames into **segments** and exporting each segment as an XES **event**.

#### NA semantics

 In IKEA ASM, `NA` denotes **“no action”** (no assembly-specific activity is performed). In our process-mining interpretation, `NA` corresponds to a **non-existent event**:

- We **keep** `NA` as a possible *prediction* (it can appear in `probs_json` for any event).
- For control-flow statistics (trace length, variants, alphabet), events with highest probability to be `NA` are treated as **not contributing**
  to the trace (see `pred_merged_log_stats.py --ignore_na_in_control_flow`).

 **How NA is encoded in XES.** In the exported XES, each NA event is explicitly marked by the **event attribute**
 `na:is_no_event=true` (see `action/export_uncertain_xes_from_segments.py`). This makes NA events easy to identify for downstream
 filtering/statistics, even if the activity label is not used for control-flow.

 In addition, you can control how NA interacts with process discovery tools via `--na_handling`:
 - `--na_handling keep` (default): NA events get `concept:name=NA`. Many tools treat this as a regular activity, so NA can appear in
   discovered models/DFGs unless you filter it out.
 - `--na_handling omit_concept_name`: NA events are still kept (including their timestamps and `probs_json`), but the exporter **omits**
   `concept:name` for those events. Many control-flow analyses rely on `concept:name` to build the activity sequence; omitting it reduces
   the chance that NA is treated as an activity, while still retaining NA events as “no-event markers” for uncertainty/statistics.

#### Exporting the test-split logs (paper setting)

This pipeline uses the released **pretrained model outputs** + **action annotations**.
Download the two zip files from the official links above (**Pretrained models - action** and **Annotations - action**) and either:

- place them in the repository root as `pt_models_action_recognition.zip` and `action_annotations.zip`, or
- pass their paths via `--pretrained_zip` and `--annotations_zip`.

1) Export per-model `frames.csv` + `segments_gt.csv` + `xes_uncertain_gt.xes` for all models found in the pretrained bundle:

```bash
python3 action/run_pretrained_export_pipeline.py \
  --dataset_name ikea_asm \
  --split test \
  --pretrained_zip "pt_models_action_recognition.zip" \
  --annotations_zip "action_annotations.zip" \
  --work_dir "_tmp_export_work" \
  --out_root "uncertain_event_data"
```

This creates (one folder per model) under:
`uncertain_event_data/ikea_asm/split=test/`

2) (Optional) Create “no-GT-NA” derived logs (remove only events with `gt:label=NA`):

```bash
python3 action/filter_xes_remove_na_events.py \
  --input     "uncertain_event_data/ikea_asm/split=test" \
  --output    "uncertain_event_data/ikea_asm/split=test_no_gt_na" \
  --recursive
```

3) Create pred-merged logs (activity = prediction, GT attributes omitted):

```bash
python3 action/run_pred_merged_export_from_frames_folder.py \
  --root "uncertain_event_data/ikea_asm/split=test" \
  --na_handling keep
```

#### Where outputs are stored

Each exported model folder has:

- `frames.csv`: one row per frame (GT label, predicted top-1 label, and `probs_json`)
- `segments_gt.csv`: one row per **GT-merged** segment (stores `avg_probs_json`)
- `xes_uncertain_gt.xes`: one XES event per GT segment (`concept:name` = GT label)
- `manifest.json`: provenance + portable relative paths

Pred-merged export additionally writes:

- `segments_pred.csv`
- `xes_uncertain_pred_merged.xes`

#### Example: what `frames.csv` and `segments_gt.csv` look like

`frames.csv` (one row per timestamp / frame; `probs_json` shortened here):

```text
case_id,case_name,timestamp,gt_label,gt_label_name,pred_label,pred_label_name,probs_json
0,Example/0001,0,21,pick up leg,0,NA,{"NA":0.29,"pick up leg":0.11,"tighten leg":0.07,...}
0,Example/0001,1,21,pick up leg,21,pick up leg,{"NA":0.22,"pick up leg":0.31,"tighten leg":0.05,...}
```

`segments_gt.csv` (consecutive identical **GT** labels merged; `avg_probs_json` shortened here):

```text
case_id,case_name,start_timestamp,end_timestamp,duration_frames,gt_label,gt_label_name,pred_label,pred_label_name,avg_probs_json
0,Example/0001,0,91,92,21,pick up leg,0,NA,{"NA":0.34,"pick up leg":0.09,"tighten leg":0.05,...}
0,Example/0001,92,144,53,1,align leg screw...,8,flip table top,{"NA":0.17,"align leg screw...":0.05,...}
```

#### Statistics used in the paper

- **Frame-wise accuracy (incl. NA)** on the test split (computed from `frames.csv`):

```bash
python3 frame_accuracy_stats.py --split_dir "uncertain_event_data/ikea_asm/split=test"
```

- **Uncertainty + control-flow stats** for pred-merged logs (with NA ignored in trace metrics):

```bash
python3 pred_merged_log_stats.py \
  --input "uncertain_event_data/ikea_asm/split=test" \
  --recursive \
  --ignore_na_in_control_flow
```

- **Event-level accuracy + entropy/top-1** for GT-aligned XES (if needed):

```bash
python3 uncertain_log_stats.py \
  --input "uncertain_event_data/ikea_asm/split=test" \
  --recursive \
  --include_na_in_accuracy
```

#### Code pointers

- `action/run_pretrained_export_pipeline.py`: zip bundle → `frames.csv` → `segments_gt.csv` → `xes_uncertain_gt.xes`
- `action/run_gt_aligned_export_pipeline.py`: same export, but starting from a single `results/` directory (`pred.npy` + `action_segments.json`)
- `action/run_pred_merged_export_from_frames_folder.py`: `frames.csv` → `segments_pred.csv` → `xes_uncertain_pred_merged.xes`
- `action/export_uncertain_xes_from_segments.py`: low-level XES writer (adds XES **classifiers** + **globals** for tool compatibility)
- `action/filter_xes_remove_na_events.py`: remove only `gt:label = NA` events (kept events unchanged)
- `frame_accuracy_stats.py`: frame-wise accuracy (incl./excl. NA) computed on `frames.csv`
- `pred_merged_log_stats.py`: trace/control-flow + uncertainty stats for pred-merged logs (supports ignoring NA for trace length)

### Benchmarks

We provide several benchmarks:

* Action recognition
* Pose Estimation
* Part segmentation and tracking

Please refer to the `README.md` file in the individual benchmark dirs for further details on training, testing and evaluating the different benchmarks (action recognition, pose estiamtion, intance segmentation, and part tracking).
Make sure to download the relevant pretrained models from the links above.

### Repository hygiene (what is typically not committed)

Generated outputs and large data extracts should not be pushed to GitHub:

- `uncertain_event_data/` (derived event logs)
- extracted video datasets (e.g. `*.avi`, extracted frames)
- scratch folders like `_tmp_*` and pipeline runs

This repository’s `.gitignore` is set up accordingly.

### License

Our code is released under MIT license (see `LICENCE.txt` file).
