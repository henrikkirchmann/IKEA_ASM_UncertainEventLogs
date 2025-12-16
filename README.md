# IKEA Assembly Dataset

This repo contains code for the "IKEA assembly dataset". This is a dev repo, after cleanup, it will be publicly available on Github. 


[Link to google drive video dataset](https://drive.google.com/file/d/1X0So9X_LQZQcCGC5DagMp3S1qy3I_XXn/view?usp=sharing) ~240GB

[Link to project website](https://ikeaasm.github.io/)

**The IKEA ASM Dataset**: Understanding People Assembling Furniture through Actions, Objects and Pose
---


### Introduction
This is the code for processing the IKEA assembly dataset.

This work will be presented in WACV 2021. 

Abstract: 

The availability of a large labeled dataset is a key requirement for applying deep learning methods to solve various computer vision tasks. In the context of understanding human activities, existing public datasets, while large in size, are often limited to a single RGB camera and provide only per-frame or per-clip action annotations. To enable richer analysis and understanding of human activities, we introduce IKEA ASM---a three million frame, multi-view, furniture assembly video dataset that includes depth, atomic actions, object segmentation, and human pose. Additionally, we benchmark prominent methods for video action recognition, object segmentation and human pose estimation tasks on this challenging dataset. The dataset enables the development of holistic methods, which integrate multi-modal and multi-view data to better perform on these tasks.

### Citation
If you find this dataset useful in your research, please cite our work:

[Preprint](https://arxiv.org/abs/2007.00394):

    @article{ben2020ikea,
      title={The IKEA ASM Dataset: Understanding People Assembling Furniture through Actions, Objects and Pose},
      author={Ben-Shabat, Yizhak and Yu, Xin and Saleh, Fatemeh Sadat and Campbell, Dylan and Rodriguez-Opazo, Cristian and Li, Hongdong and Gould, Stephen},
      journal={arXiv preprint arXiv:2007.00394},
      year={2020}
    }

[WACV2021](https://openaccess.thecvf.com/content/WACV2021/html/Ben-Shabat_The_IKEA_ASM_Dataset_Understanding_People_Assembling_Furniture_Through_Actions_WACV_2021_paper.html): 

    @inproceedings{ben2021ikea,
      title={The ikea asm dataset: Understanding people assembling furniture through actions, objects and pose},
      author={Ben-Shabat, Yizhak and Yu, Xin and Saleh, Fatemeh and Campbell, Dylan and Rodriguez-Opazo, Cristian and Li, Hongdong and Gould, Stephen},
      booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
      pages={847--859},
      year={2021}
    }
    
### Installation
Please first download the dataset using the provided links: 
[Full dataset download](https://drive.google.com/drive/folders/1xkDp--QuUVxgl4oJjhCDb2FWNZTkYANq?usp=sharing)

Alternatively, you can download only the relevant parts:  
* [utility files](https://drive.google.com/file/d/11D7d8XBRg-CPIxMroviQEaaMhw3EaGnB/view?usp=sharing)
* [camera parameters](https://drive.google.com/file/d/1BRq9HJQeEJFbhnCwGwY3eXe1587TybCe/view?usp=sharing)
* [Data - RGB top view](https://drive.google.com/file/d/1CFOH-W-6N50AVA_NqHnm06GUsfpcka0L/view?usp=sharing)
* [Data - RGB multi-view](https://drive.google.com/file/d/1eCbrIuw--16xCmI3RtBhRJ-r9K_FVkL6/view?usp=sharing)
* [Data - Depth](https://drive.google.com/file/d/18FKRSzoUiO3EV_J2WmQyvmPGiHJcH28S/view?usp=sharing)
* [Annotations - action](https://drive.google.com/file/d/1SwBNLViktSpk99jhh3sMXVGTMVr6tpju/view?usp=sharing)
* [Annotations - pose](https://drive.google.com/file/d/1RE7Ya1gwogqJtJIi5WeYOH4_Cs1RuTx7/view?usp=sharing)
* [Annotations - segmetnation and tracking](https://drive.google.com/file/d/1_jRCcLAz9zhXTnNnslBUJcu2sZjp9dVV/view?usp=sharing)
* [Pretrained models - action](https://drive.google.com/file/d/1QksK_Uvty6pTYoGmBGWYYG3scvM_NX2X/view?usp=sharing)
* [Pretrained models - pose](https://drive.google.com/file/d/1SMoYC-PTHr6Y2StKKT8j_-gSYcwhTHKb/view?usp=sharing)
* [Pretrained models - segmentation and tracking](https://drive.google.com/file/d/1lLNiWU6ILFCgg104FDwWvRMV0iQaGKyp/view?usp=sharing)

 
After downloading the video data, extract the individual frames using `./toolbox/extract_frames_from_videos.py`
For further processing of the data refer to the individual benchmarks `README.md` files.

For depenencies see `requirements.txt`.

### Notes for this fork (process mining / uncertain event logs)
This workspace includes additional scripts and generated artifacts to export **uncertain event logs** (CSV + XES) and to compute
log-level statistics (including **event accuracy** and **uncertainty**). These additions are not part of the original upstream repo.

#### Folder layout (what’s in here, and how it relates to XES)
This repo has two kinds of folders:
- **Inputs**: annotations, model outputs, dataset extracts (often large; usually not committed)
- **Outputs**: exported process-mining artifacts (CSV/XES) under `uncertain_event_data/`

Key folders:

- **[`_annotations_extract/`](./_annotations_extract/)**
  - **Contains**: `gt_segments.json`, `gt_action.npy`, `ikea_annotation_db_full` (sqlite DB).
  - **Role**: ground truth + label schema.
    - `gt_segments.json` provides the **canonical label list** used to name classes and to key `probs_json`.
    - `ikea_annotation_db_full` is the authoritative GT source used by the dataloader (and DB-based alignment).
  - **XES relation**: used to produce the (only supported) **GT-aligned** XES where event `concept:name` is the real GT label.

- **[`_depth_dataset_extract/`](./_depth_dataset_extract/)**
  - **Contains**: a local extracted dataset root (e.g. `ANU_ikea_dataset_video/`) with depth videos (`*.avi`) and `indexing_files/`.
  - **Role**: enables running the repo’s action-recognition inference on the depth zip.
  - **XES relation**: enables generating depth-model XES by first producing a `results/` folder (`pred.npy` + `action_segments.json`).

- **[`action/`](./action/)**
  - **Contains**: the exporter scripts that generate uncertain event logs.
  - **Important scripts**:
    - `action/align_pred_results_with_db_gt.py`: takes an existing model `results/` dir and produces an *aligned* `pred.npy`
      that includes `gt_labels` (from the sqlite DB) + a rebuilt `action_segments.json` (correct order).
    - `action/run_gt_aligned_export_pipeline.py`: turns a `results/` dir into the default **GT-aligned** export:
      - `frames.csv` (per-frame GT + prediction + `probs_json`)
      - `segments_gt.csv` (consecutive GT runs aggregated; per-segment `avg_probs_json`)
      - `xes_uncertain_gt.xes` (one event per GT segment; event includes `gt:*`, `pred:*`, `probs_json`)
    - `action/filter_xes_remove_na_events.py`: derives a **no-NA** XES by removing `gt:label == NA` events and renormalizing
      `probs_json` (excluding NA) while updating `pred:label`/`pred:label_id`.
    - `action/export_uncertain_xes_from_segments.py`: low-level XES writer (segment CSV → XES).

- **[`uncertain_event_data/`](./uncertain_event_data/)**
  - **Contains**: the “final” exported artifacts you typically open in process mining tools.
  - **Layout**:
    - `uncertain_event_data/<dataset_name>/split=<split>/labels/` (label schema)
    - `uncertain_event_data/<dataset_name>/split=<split>/model=<...>/`
      - `frames.csv`: per-frame records with `gt_label_name`, `pred_label_name`, and `probs_json`
      - `segments_gt.csv`: GT-segment events with `avg_probs_json`
      - `xes_uncertain_gt.xes`: the event log (XES)
      - `manifest.json`: provenance (where inputs came from)
  - **XES variants you may see**:
    - **GT-aligned (default and only export mode)**: `concept:name` is GT, `pred:*` is model prediction, and `probs_json` stores uncertainty.
    - **No-NA derived**: file name like `xes_uncertain_gt__no_na.xes` created by the filter script (NA events removed, probs renormalized).

- **[`uncertain_log_stats.py`](./uncertain_log_stats.py)**
  - **Role**: evaluates an XES directly (no pm4py needed): event accuracy (incl/excl NA), NA correctness, uncertainty from `probs_json`.

#### What NOT to commit
Some folders in this workspace are **local extracts** or **generated run artifacts** and should not be pushed to GitHub:
- `_depth_dataset_extract/ANU_ikea_dataset_video/**/*.avi` (large depth videos)
- `_depth_dataset_extract/ANU_ikea_dataset_video/ikea_annotation_db_full` (copied sqlite DB)
- `_run_depth_zip_all371/`, `_pipeline_run_*/`, `_tmp_*/` (generated outputs / scratch)

These are ignored via `.gitignore`.

#### How to regenerate `_depth_dataset_extract/.../indexing_files/` on setup
After you download and extract the **depth dataset** locally (so you have an extracted dataset root like
`_depth_dataset_extract/ANU_ikea_dataset_video/`), regenerate the expected split/index files by running:

```bash
python3 action/prepare_depth_zip_dataset_all.py \
  --dataset_root "_depth_dataset_extract/ANU_ikea_dataset_video" \
  --db_src "_annotations_extract/ikea_annotation_db_full" \
  --camera dev3
```

This will:
- copy the sqlite DB into the dataset root (needed for the dataloader)
- write `indexing_files/atomic_action_list.txt`, `action_object_relation_list.txt`
- write `indexing_files/test_cross_env.txt` containing all depth videos found on disk
- write an empty `indexing_files/train_cross_env.txt`

#### Example: frame CSV → segmented CSV → XES
After running the export pipeline, each model folder under `uncertain_event_data/<dataset>/split=<split>/model=.../` contains:

- `frames.csv`: **one row per frame/timestamp**
- `segments_gt.csv`: **one row per GT segment** (consecutive frames with same GT label are merged)
- `xes_uncertain_gt.xes`: **event log**, one event per segment (activity = GT label via `concept:name`)

**About `NA`**: In this dataset, the label `NA` means **“no action”** (no special atomic assembly action is happening in those frames).
It is the default/background class used for unlabeled or transitional frames.

Concrete example (depth model on depth-zip, split=all):

- `uncertain_event_data/ikea_asm_depth_zip/split=all/model=clip_based__i3d__depth__depth/frames.csv`
- `uncertain_event_data/ikea_asm_depth_zip/split=all/model=clip_based__i3d__depth__depth/segments_gt.csv`
- `uncertain_event_data/ikea_asm_depth_zip/split=all/model=clip_based__i3d__depth__depth/xes_uncertain_gt.xes`

`frames.csv` (per-frame) key columns:
- `case_id`, `case_name`: which assembly video / trace
- `timestamp`: frame index
- `gt_label`, `gt_label_name`: GT class id + name
- `pred_label`, `pred_label_name`: predicted class id + name
- `probs_json`: probability distribution over all actions for that frame

Example `frames.csv` rows (shortened `probs_json` for readability):

```text
case_id,case_name,timestamp,gt_label,gt_label_name,pred_label,pred_label_name,probs_json
0,Lack_TV_Bench/0006_white_table_07_03_2019_08_21_16_37,0,21,pick up leg,0,NA,{"NA":0.29228806495666504,"align leg screw with table thread":0.07027541846036911,"align side panel holes with front panel dowels":0.0259142…}
0,Lack_TV_Bench/0006_white_table_07_03_2019_08_21_16_37,1,21,pick up leg,0,NA,{"NA":0.2841722071170807,"align leg screw with table thread":0.0745786651968956,"align side panel holes with front panel dowels":0.026076121…}
```

Readable toy example (50 timestamps → 5 GT segments)

This is a **small, human-readable illustration** of how `frames.csv` and `segments_gt.csv` relate. It uses **imaginary data**:
- 1 case (`case_id=0`, `case_name=Example/0001`)
- timestamps `0..49`
- 5 GT segments of length 10 frames each
- probabilities omitted to keep the table readable

Toy `frames.csv` (50 rows):

| timestamp | gt_label_name | pred_label_name |
|---:|---|---|
| 0 | pick up leg | NA |
| 1 | pick up leg | NA |
| 2 | pick up leg | pick up leg |
| 3 | pick up leg | pick up leg |
| 4 | pick up leg | pick up leg |
| 5 | pick up leg | pick up leg |
| 6 | pick up leg | NA |
| 7 | pick up leg | pick up leg |
| 8 | pick up leg | pick up leg |
| 9 | pick up leg | pick up leg |
| 10 | align leg screw with table thread | align leg screw with table thread |
| 11 | align leg screw with table thread | align leg screw with table thread |
| 12 | align leg screw with table thread | align leg screw with table thread |
| 13 | align leg screw with table thread | tighten leg |
| 14 | align leg screw with table thread | align leg screw with table thread |
| 15 | align leg screw with table thread | align leg screw with table thread |
| 16 | align leg screw with table thread | align leg screw with table thread |
| 17 | align leg screw with table thread | align leg screw with table thread |
| 18 | align leg screw with table thread | align leg screw with table thread |
| 19 | align leg screw with table thread | align leg screw with table thread |
| 20 | tighten leg | tighten leg |
| 21 | tighten leg | tighten leg |
| 22 | tighten leg | tighten leg |
| 23 | tighten leg | tighten leg |
| 24 | tighten leg | tighten leg |
| 25 | tighten leg | align leg screw with table thread |
| 26 | tighten leg | tighten leg |
| 27 | tighten leg | tighten leg |
| 28 | tighten leg | tighten leg |
| 29 | tighten leg | tighten leg |
| 30 | flip table top | flip table top |
| 31 | flip table top | flip table top |
| 32 | flip table top | flip table top |
| 33 | flip table top | flip table top |
| 34 | flip table top | flip table top |
| 35 | flip table top | flip table top |
| 36 | flip table top | tighten leg |
| 37 | flip table top | flip table top |
| 38 | flip table top | flip table top |
| 39 | flip table top | flip table top |
| 40 | NA | NA |
| 41 | NA | NA |
| 42 | NA | NA |
| 43 | NA | pick up leg |
| 44 | NA | NA |
| 45 | NA | NA |
| 46 | NA | NA |
| 47 | NA | NA |
| 48 | NA | NA |
| 49 | NA | NA |

Toy `segments_gt.csv` derived from the table above (5 segments):

| segment_idx | start_timestamp | end_timestamp | duration_frames | gt_label_name | pred_label_name (from avg_probs_json argmax) |
|---:|---:|---:|---:|---|---|
| 1 | 0 | 9 | 10 | pick up leg | pick up leg |
| 2 | 10 | 19 | 10 | align leg screw with table thread | align leg screw with table thread |
| 3 | 20 | 29 | 10 | tighten leg | tighten leg |
| 4 | 30 | 39 | 10 | flip table top | flip table top |
| 5 | 40 | 49 | 10 | NA | NA |

`segments_gt.csv` (per-segment) key columns:
- `start_timestamp`, `end_timestamp`, `duration_frames`: segment boundaries in frame indices
- `gt_label`, `gt_label_name`: GT label for the whole segment
- `pred_label`, `pred_label_name`: argmax of the **average** probabilities over the segment
- `avg_probs_json`: averaged probability distribution over the segment

Example `segments_gt.csv` rows (shortened `avg_probs_json` for readability):

```text
case_id,case_name,start_timestamp,end_timestamp,duration_frames,gt_label,gt_label_name,pred_label,pred_label_name,avg_probs_json
0,Lack_TV_Bench/0006_white_table_07_03_2019_08_21_16_37,0,91,92,21,pick up leg,0,NA,{"NA":0.33806702753771906,"align leg screw with table thread":0.049669797863791006,"align side panel holes with front panel dowels":0.017750…}
0,Lack_TV_Bench/0006_white_table_07_03_2019_08_21_16_37,92,144,53,1,align leg screw with table thread,8,flip table top,{"NA":0.16960671501901914,"align leg screw with table thread":0.04731743419015745,"align side panel holes with front panel dowels":0.0022158…}
```

`xes_uncertain_gt.xes` (event log):
- each `<trace>` corresponds to one `case_name`
- each `<event>` corresponds to one row in `segments_gt.csv`
- activity name = `<string key="concept:name" ...>` which is set to the GT label name (non-NA events)
- uncertainty stored as `<string key="probs_json" ...>` (a JSON dict label→prob)

#### Example: remove NA events and renormalize probabilities (Disco-friendly “no-NA” log)
To derive a log without NA events **and** renormalize the probability distribution so that **NA cannot be predicted**:

```bash
python3 action/filter_xes_remove_na_events.py \
  --input  "uncertain_event_data/ikea_asm_depth_zip/split=all/model=clip_based__i3d__depth__depth/xes_uncertain_gt.xes" \
  --output "uncertain_event_data/ikea_asm_depth_zip/split=all/model=clip_based__i3d__depth__depth/xes_uncertain_gt__no_na.xes"
```

What this does for every kept (non-NA) event:
- removes the `NA` entry from `probs_json`
- rescales the remaining probabilities so they sum to 1 again
- sets `pred:label`/`pred:label_id` to the **new argmax** of the renormalized distribution (so NA is never predicted)

#### New scripts added in this fork
- `uncertain_log_stats.py`: compute event-log stats from XES (event accuracy, NA correctness, uncertainty, etc.)
- `action/run_gt_aligned_export_pipeline.py`: export GT-aligned (default) `frames.csv` → `segments_gt.csv` → `xes_uncertain_gt.xes`
- `action/align_pred_results_with_db_gt.py`: attach DB-derived GT labels to an existing `results/pred.npy` and rebuild `action_segments.json`
- `action/filter_xes_remove_na_events.py`: create a derived XES with NA events removed, and renormalize `probs_json` (excluding NA) while updating `pred:label`/`pred:label_id` accordingly

### Benchmarks
We provide several benchmarks: 
* Action recognition
* Pose Estimation
* Part segmentation and tracking

Please refer to the `README.md` file in the individual benchmark dirs for further details on training, testing and evaluating the different benchmarks (action recognition, pose estiamtion, intance segmentation, and part tracking).
Make sure to download the relevant pretrained models from the links above.

### Running this fork on another machine (Cursor/GitHub quickstart)
This repo is now cleaned to be **code-only**: large datasets, extracted artifacts, and exported logs are not included.

#### External inputs you must provide locally
To recreate event logs you need:
- **An extracted dataset root** (RGB/depth videos) in the original IKEA ASM layout.
- **The sqlite annotation DB**: `ikea_annotation_db_full`
- **Label schema JSON**: `gt_segments.json` (used to map label ids → label names and to key `probs_json`)
- Either:
  - **Model inference outputs**: a `results/` directory containing `pred.npy` + `action_segments.json`, or
  - The ability to run inference locally (dataset + checkpoint) to generate that `results/` directory.

#### Minimal end-to-end (from existing model outputs)
If you already have `results/` from a model (predicted probabilities per frame):

1) (Recommended) Align results with DB-derived GT + correct ordering:

```bash
python3 action/align_pred_results_with_db_gt.py \
  --dataset_root "<PATH_TO_EXTRACTED_DATASET_ROOT>" \
  --pred_results_dir "<PATH_TO_RESULTS_DIR>" \
  --out_results_dir "<PATH_TO_OUTPUT_ALIGNED_RESULTS_DIR>" \
  --camera dev3 \
  --frame_skip 1 \
  --frames_per_clip 64 \
  --dataset_mode vid \
  --input_type depth
```

2) Export GT-aligned `frames.csv` → `segments_gt.csv` → `xes_uncertain_gt.xes`:

```bash
python3 action/run_gt_aligned_export_pipeline.py \
  --dataset_name ikea_asm_depth_zip \
  --split all \
  --pred_results_dir "<PATH_TO_OUTPUT_ALIGNED_RESULTS_DIR>" \
  --gt_segments_json "<PATH_TO_GT_SEGMENTS_JSON>" \
  --work_dir "<SCRATCH_WORK_DIR>" \
  --out_root "<OUTPUT_ROOT_DIR>" \
  --use_pred_gt
```

#### Create a GitHub repo (recommended workflow)
From the repo root:

```bash
git init
git add .
git commit -m "Initial commit: GT-aligned uncertain event log export"
```

Create an empty repo on GitHub (via the web UI), then:

```bash
git remote add origin <YOUR_GITHUB_REPO_GIT_URL>
git branch -M main
git push -u origin main
```

### License
Our code is released under MIT license (see `LICENCE.txt` file).
