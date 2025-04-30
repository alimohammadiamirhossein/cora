# Arguments
This is a description to provide details about arguments of Cora. It is implemented by Amir Alimohammadi.

## 📂 Directory layout
```
Cora
├── main.py
├── scripts
|   ├── edit.sh                       -- script for image editing
|   ├── config.sh                     -- configuration settings
|   ├── prompts
│   │   ├── p.json                    -- image editing prompts, alpha/beta, and masks 
|   ├── ...  
```

---

## ⚙️ Edit parameters (used by `scripts/config.sh`)
You can easily configure different components by setting parameters in `config.sh`, allowing you to test, enable, or disable various features.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--num_of_timesteps` | int | `4` | Number of diffusion steps. Everything is tuned for 4 steps, but you can experiment with other counts (see `scripts/steps_mode`). |
| `--output_dir` | string | — | Path to the output folder. *Example*: `--output_dir=result` |
| `--prompts_files` | string | `scripts/prompts/p.json` | JSON file containing image paths, prompts, and α/β values. |
| `--support_new_object` | boolean | `true` | Enables content-adaptive interpolation. Disabling it makes the model struggle when generating new content (e.g., adding a hat) with small α. |
| `--structural_alignment` | boolean | `true` | Enables the structural alignment module. Setting it to `false` disables structural alignment, and the model will ignore β during editing. |
| `--w1` | float | `1.9` | Pseudo-guidance weight (from **TurboEdit**). It behaves like guidance scale: higher values strengthen the prompt. |
| `--mode` | string | `slerp_dift` | Attention strategy:<br>• `slerp_dift` — SLERP with DiFT correspondence<br>• `slerp` — naïve SLERP<br>• `lerp_dift` — linear interp. + DiFT<br>• `lerp` — naïve linear interp.<br>• `normal` — attention only on target (source ignored)<br>• `concat` — concatenate source & target<br>• `masa` — use source as key/value.<br/>See Table 2 of the paper for ablation results. |
| `--movement_intensifier` | float | `0.2` | Strength of non-rigid editing (requires a mask). Details in <https://arxiv.org/abs/2407.17850>. |
| `--dift_timestep` | int | `400` | First timestep at which DiFT correction is applied. For the last three steps, set e.g. `700`; for the last two (slightly better), keep `400`. |
| `--apply_dift_correction` | boolean | `true` | Toggles Correspondence-Aware Latent Correction (CLC). Turn off (with the same seed) to observe its effect. |

---


## 📝 Using a JSON file for editing (`scripts\prompts\p.json`)

To perform edits, you need to prepare a JSON file that specifies:
- the source image,
- the source prompt,
- the target prompts,
- the alpha/beta blending values,
- and optionally, specific regions to edit (masks).

Each entry in the JSON should follow this structure:
- `src_prompt`: Prompt describing the source image.
- `tgt_prompt`: A list of one or more prompts for the desired edit.
- `alpha`: (optional): Controls how much of the original appearance is preserved (0 = fully original content, 1 = fully new content).
- `beta` (optional): Controls how much of the original structure is preserved.
- `masks` (optional): List of regions where editing should be applied, each specified by:
  - `start_point`: the top-left `(x, y)` coordinate of the region to edit.
  - `end_point`: the bottom-right `(x, y)` coordinate of the region to edit.

Here is a sample JSON you can use:

```json
{
  "pearl.png": {
    "src_prompt": "a europian girl",
    "tgt_prompt": ["an asian girl"],
    "alpha": 0.0,
    "beta": 0.2,
    "masks":[
      {
        "start_point": [108, 18],
        "end_point": [501, 512]
      }
    ]
  },
  "dolphin.jpg": {
    "src_prompt": "a photo of a dolphin",
    "tgt_prompt": ["a photo of a shark"],
    "alpha": 0.6,
    "beta": 0.4,
  }
}
```

## 🖼️ Extracting Masks Interactively

If you want to define mask regions manually, you can use our interactive bounding box tool. This tool allows you to click and drag on an image to define the area to edit.

To launch the tool, simply run:

```bash
python ./visualization/draw_box.py
```

### How it works:

- The image will open in a window.
- Click and hold the left mouse button to start drawing a bounding box.
- Drag the mouse to define the area you want to edit.
- Release the mouse button to finalize the box.
- The `start_point` and `end_point` coordinates will automatically be printed in the terminal.
- Copy these coordinates and paste them into your JSON file under the `masks` field.

> 📌 **Tip:** Images are automatically resized to `(512, 512)` in the script to ensure consistency across edits.

### 🎯 Why Use Masks?

We highly recommend providing a mask for the region you want to edit. This allows Cora to better preserve the background and improve overall quality. Unlike some other methods that apply masking via cross-attention, Cora does not natively use cross-attention-based masking. 

> ✅ **Recommendation:** Always use a mask when editing localized content (e.g., adding accessories, changing objects, or modifying small regions).

