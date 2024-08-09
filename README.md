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
```
ns-train splatfacto scannet-data --data nvsmask3d/data/scene0000_00_  # this is old scannet dataparser
ns-train nvsmask3d scannet --data nvsmask3d/data/scene_example # this is new scannet dataparser modified in this repo
ns-train nvsmask3d replica --data nvsmask3d/data/Replica # train on replica data
ns-train splatfacto --vis viewer+wandb colmap --data nvsmask3d/data/scene0011_00/colmap 
```
## View
```
ns-viewer nvsmask3d --load_config outputs/nvsmask3d_whole_data_0.9_train_ratio/nvsmask3d/2024-08-01_144608/config.yml
ns-viewer nvsmask3d --load_config outputs/scene0011_00/nvsmask3d/2024-08-07_220010/config.yml
```

## Evaluation 
```
ns-eval for_ap --load_config outputs/nvsmask3d_whole_data_0.9_train_ratio/nvsmask3d/2024-08-01_144608/config.yml # full
ns-eval for_ap --load_config outputs/fast/nvsmask3d/2024-08-07_175651/config.yml  #2000
ns-eval for_ap --load_config outputs/fast/nvsmask3d/2024-08-07_181023/config.yml #6000
ns-eval for_ap --load_config outputs/scene0011_00/nvsmask3d/2024-08-07_220010/config.yml
ns-eval for_ap --load_config outputs/scene0011_00/nvsmask3d/2024-08-08_132932/config.yml   #sparse point cloud initialization
ns-eval for_ap --load_config outputs/scene0011_00/nvsmask3d/2024-08-08_165724/config.yml #sparse point cloud initialization+densify+culling  20000steps
ns-eval for_ap --load_config outputs/unnamed/splatfacto/2024-08-08_210343/config.yml #colmap
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

# Datasets
### Replica
Download dataset with: `python nvsmask3d/data/download_replica.py`
