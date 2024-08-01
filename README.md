## Registering with Nerfstudio
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html). Clone or fork this repository and run the commands:

```
conda activate nerfstudio
cd nerfstudio-method-template/
pip install -e .
ns-install-cli
```
## info
20% images used in dataparser
tran test split = 1
test_mode = train

## Running the new method
This repository creates a new Nerfstudio method named "method-template". To train with it, run the command:
```
ns-train method-template --data [PATH]
ns-train splatfacto scannet-data --data nvsmask3d/data/scene0000_00_ 
ns-train nvsmask3d --data nvsmask3d/data/scene_example
ns-train nvsmask3d --data nvsmask3d/data/scene0000_00_ --vis viewer+wandb
```
## View
```
ns-viewer nvsmask3d --load_config outputs/scene0000_00_/nvsmask3d/2024-07-25_144828/config.yml
ns-viewer nvsmask3d --load_config outputs/scene0000_00_/nvsmask3d/2024-07-29_211122/config.yml
ns-viewer nvsmask3d --load_config outputs/nvsmask3d_whole_data_0.9_train_ratio/nvsmask3d/2024-08-01_144608/config.yml
```

## Evaluation 
```

```
## Single scene predction path
```
/home/wangs9/junyuan/openmask3d/output/2024-07-23-11-44-44-scene0000_00_/scene0000_00__masks.pt
```

[INFO] Shape of instance masks: (1990518, 166)

1) Start local server: 
    cd /home/wangs9/junyuan/openmask3d/openmask3d/saved/scene0000_00_/visualizations/scene0000_00_; python -m http.server 6008
2) Open in browser:
    http://0.0.0.0:6008






      File "/home/wangs9/junyuan/nerfstudio-nvsmask3d/nvsmask3d/nvsmask3d_model.py", line 236, in get_outputs
    opacities_masked = opacities_crop[mask_indices]
IndexError: The shape of the mask [1990518] at index 0 does not match the shape of the indexed tensor [739790, 1] at index 0