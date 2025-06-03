
<div align="center">

# Cora: Correspondence-aware Image Editing Using Few-Step Diffusion

[Amir Alimohammadi](https://alimohammadiamirhossein.github.io/)<sup>*1</sup> &nbsp;&nbsp;
[Aryan Mikaeili](https://aryanmikaeili.github.io/)<sup>*1</sup> &nbsp;&nbsp;
[Sauradip Nag](https://sauradip.github.io/)<sup>1</sup> &nbsp;&nbsp;
[Negar Hassanpour](https://webdocs.cs.ualberta.ca/~hassanpo/)<sup>2</sup>  
[Andrea Tagliasacchi](https://taiya.github.io/)<sup>1,3,4</sup> &nbsp;&nbsp;
[Ali Mahdavi Amiri](https://www.sfu.ca/~amahdavi)<sup>1</sup>

<sup>1</sup> **Simon Fraser University** &nbsp;&nbsp;
<sup>2</sup>Huawei &nbsp;&nbsp;
<sup>3</sup>University of Toronto &nbsp;&nbsp;
<sup>4</sup>Google DeepMind  

*\* indicates equal contribution*  

**Accepted at SIGGRAPH&nbsp;2025**

</div>


<h3 align="center">
  <a href="https://arxiv.org/abs/2505.23907" target='_blank'>Paper</a> |
  <a href="https://cora-edit.github.io/" target='_blank'>Project Page</a> 
</h3>
</div>

**Cora** is a new image editing method that enables flexible and accurate edits, such as pose changes, object insertions, and background swaps, using only four diffusion steps. Unlike other fast methods that often produce visual artifacts, Cora uses *semantic correspondences* between the original and edited images to preserve structure and appearance where necessary. It is fast, controllable, delivers high-quality edits, and requires no additional training.


<div align="center">
<table>
<tr>
    <td><img src="./assets/teaser.png" width="100%"/></td>
</tr>
</table>
</div>

## Why Cora?

* **Few-step editing** – delivers high-quality results in just 4 diffusion steps, making it fast and memory-efficient.  
* **Structure-aware** – *Correspondence-aware Latent Correction* (CLC) aligns noise terms to the target layout, eliminating ghosting artifacts.  
* **Appearance control** – *Attention Interpolation* lets you blend source appearance with the prompt via a controllable parameter (α).  
* **Structure control** – *Query Matching* maintains or relaxes spatial layout via a controllable parameter (β).  
* **Plug-and-play** – built on 🤗 *Diffusers*; works with any Stable-Diffusion-compatible model.  

If Cora helps your research, please consider starring ⭐ the repo!


# Overview

The main parts of the framework are as follows:

```
Cora
├── main.py                            
├── model                    
│   ├── directional_attentions.py     -- attention for controlling appearance and layout
|   ├── modules
│   │   ├── dift_utils.py             -- feature alignment and patch-based latent matching utilities        
│   │   ├── new_object_detection.py   -- new object detection for content-adaptive interpolation            
│   │   ├── ...     
|   ├── ...
├── src
|   ├── ddpm_step.py                  -- single-step denoising for DDPM schedulers      
|   ├── ddpm_inversion.py             -- correspondence-aware latent correction
├── scripts
|   ├── edit.sh                       -- script for image editing
|   ├── config.sh                     -- configuration settings
|   ├── prompts
│   │   ├── p.json                    -- image editing prompts, alpha/beta, and masks 
|   ├── ...   
├── utils                    
|   ├── args.py                       -- define, parse, and update command-line arguments
|   ├── utils.py
|   ├── ...   
├── visualization                    
|   ├── image_utils.py                -- resizing, saving images, and handling prompts.
|   ├── draw_box.py                   -- interactive bounding box drawing tool
|   ├── draw_mask.py                  -- interactive mask drawing tool
```
# Getting Started  
To get started as quickly as possible, follow the instructions in this section. This should allow you train a model from scratch, evaluate your pretrained models, and produce visualizations.  

### Dependencies
- Python 3.10+
- PyTorch == 2.5.1 **(Please make sure your pytorch version is atleast 2.1)**
- A modern NVIDIA GPU (e.g., 3090 RTX or newer)
- Hugging-Face Diffusers
- transformers == 4.43.3

### Environment Setup
You can create and activate a Conda environment like below:
```shell script
conda create -n cora_env
conda activate cora_env  
pip install --upgrade pip
```

### Requirements  
Furthermore, you just have to install all the packages you need:  
```shell script  
pip install -r requirements.txt  
```  

# Usage

### Edit

To perform inference, place your images in the `dataset` folder, create a JSON file with the source and target prompts (similar to our `dataset/dataset.json` file), and then run:

```
bash scripts/edit.sh
```

#### 🔧 Advanced Configuration & Arguments

For detailed explanations of all configurable parameters, JSON formatting, and mask extraction, see the [Arguments & Parameters Guide](./assets/ARGS_README.md).


### Gradio demo
Alternatively, if you want to experiment using [Gradio](https://www.gradio.app/)'s UI, run:
```
python app.py 
```



## Citation
If you find this project useful for your research, please use the following BibTeX entry.
```
@misc{almohammadi2025coracorrespondenceawareimageediting,
      title={Cora: Correspondence-aware image editing using few step diffusion}, 
      author={Amirhossein Almohammadi and Aryan Mikaeili and Sauradip Nag and Negar Hassanpour and Andrea Tagliasacchi and Ali Mahdavi-Amiri},
      year={2025},
      eprint={2505.23907},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.23907}, 
}
```
