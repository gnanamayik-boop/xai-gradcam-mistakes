# Do AI Explanations Reveal Model Mistakes? A Controlled Evaluation of Grad-CAM

**DS5622 Explainable AI · Track 1 (Experimental Investigation of Existing XAI Methods)**
**Team:** Gnanamayi (142301009), Megha Harsha (142301036)

## Question
Do Grad-CAM heatmaps (a) differ between correct and incorrect predictions, (b) help detect confident mistakes better than the model's own confidence, and (c) remain faithful, i.e. do the highlighted pixels matter to the model, including on mistakes?
We test four hypotheses (H1-H4, fixed before running the experiments) on a CIFAR-10 ResNet-18 using Grad-CAM at three layers, vanilla gradient saliency, a model-randomisation sanity check, deletion tests with three fill values, and a confidence-matched comparison.

## Repository contents
| Path | What it is |
|---|---|
| `xai_gradcam_project.ipynb` | The whole pipeline, end to end (training, explanations, metrics, RQ1-RQ4, sanity check, sensitivity analysis, failure cases) |
| `requirements.txt` | Python dependencies |
| `model/model.pt` | Trained ResNet-18 checkpoint used for every number in the report |
| `results/config.json` | All experiment settings (seed, training, deletion, statistics) |
| `results/summary.json` | Every key number in one file |
| `results/predictions.csv` | One row per test image: label, prediction, confidence, concentration metrics |
| `results/training_log.csv` | Per-epoch training log |
| `results/figures/` | All figures (F1-F14) |
| `results/tables/` | All tables (T1-T8) as CSV and Markdown |
| `report.pdf` | Final project report (also submitted on the LMS) |

## How to reproduce
1. Open `xai_gradcam_project.ipynb` in Google Colab and select **Runtime -> Change runtime type -> T4 GPU**.
2. **Quick test (about 10-15 min):** keep `DEBUG = True` in cell 0.2 and run all cells. This only checks that the pipeline runs; ignore its numbers.
3. **Real run (about 1.5-2.5 h):** set `DEBUG = False` and run all cells. Results are saved to Google Drive (`MyDrive/xai-gradcam-mistakes/results/`). If Colab disconnects, run all again: finished steps (model, explanations, deletion chunks) are loaded from disk.
4. **To skip training:** copy `model/model.pt` to `MyDrive/xai-gradcam-mistakes/results/model.pt` before running. The notebook then loads it instead of retraining.

Settings: seed 42, SGD (Nesterov, momentum 0.9), lr 0.1 with cosine decay, weight decay 5e-4, batch 128, 30 epochs, mixed precision, random crop and flip. See `results/config.json`.

**Note on exact numbers.** GPU training with mixed precision is not bit-for-bit deterministic, so retraining can change the model slightly (and therefore counts such as 686 errors and 172 confident mistakes). All numbers in the report come from the released `model/model.pt`.

## Dependencies
See `requirements.txt`. `grad-cam` and `captum` are used only to cross-check our own Grad-CAM and saliency implementations.

## LLM-use disclosure
We used two AI tools.
- **Claude (Anthropic):** reviewed our project proposal (suggested adding one research question), corrected code when we hit errors in the deep-learning code, helped write the code for saving checkpoints, figures and tables, polished our report draft, and explained how to use GitHub.
- **ChatGPT (OpenAI):** gave us a step-by-step workflow, helped us research how and why to do each step and the related work, generated the saliency code and a few other code sections, and helped with syntax and which libraries to use.

**Verification:** we cross-checked our Grad-CAM and saliency implementations against the `pytorch-grad-cam` and Captum libraries, ran built-in checks (identical starting probabilities at k = 0, the model-randomisation sanity check), did a quick debug run of the full pipeline, inspected every figure, and checked the methods and claims against the original research papers and other articles. Both team members reviewed the code and results and can explain them.
