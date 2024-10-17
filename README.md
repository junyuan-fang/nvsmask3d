## Registering with Nerfstudio
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html). Clone or fork this repository and run the commands:

```
conda activate nerfstudio
cd nerfstudio-method-template/
pip install -e .
ns-install-cli
```
## info
tran test split = 1
test_mode = train

for scannetpp:

path of 3d mask proposals: ```/home/fangj1/Code/nerfstudio-nvsmask3d/nvsmask3d/data/mask3d_processed_first10``` corresponding scene's names are inside ```/home/fangj1/Code/nerfstudio-nvsmask3d/nvs_sub_val.txt```.

path of scannetpp data: ```/data/scannetpp/ScannetPP/data```

### train scannetpp scene
```
ns-train nvsmask3d --experiment-name 7b6477cb95 --timestamp ""  --vis viewer scannetpp_nvsmask3d --data nvsmask3d/data/ScannetPP   --sequence 7b6477cb95 
```
## train Replica scene
```
ns-train splatfacto scannet-data --data nvsmask3d/data/scene0000_00_  # this is old scannet dataparser
ns-train nvsmask3d scannet --data nvsmask3d/data/scene_example # this is new scannet dataparser modified in this repo
ns-train nvsmask3d replica --data nvsmask3d/data/Replica # train on replica data
ns-train nvsmask3d --vis viewer replica --data nvsmask3d/data/Replica --sequence room0
ns-train splatfacto --vis viewer+wandb colmap --data nvsmask3d/data/scene0011_00/colmap 
ns-train nvsmask3d replica --data nvsmask3d/data/Replica --sequence room0
ns-train nvsmask3d --vis viewer replica_nvsmask3d --data nvsmask3d/data/replica --sequence room0 #sometimes wandb error, so use this
```


## View
```
ns-viewer nvsmask3d --load_config outputs/office0/nvsmask3d/2024-08-14_204330/config.yml # office 0

```

## Evaluation 
```

ns-eval for_ap --load_config nvsmask3d/data/replica

```
# Run evaluation in script
```
python nvsmask3d/script/eval_all.py
```
### for replica validation
```
ns-eval for_ap --load_config nvsmask3d/data/replica
```
## NVS quality eval
```
ns-eval psnr --load_config outputs/unnamed/nvsmask3d/2024-08-11_124613/config.yml
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
