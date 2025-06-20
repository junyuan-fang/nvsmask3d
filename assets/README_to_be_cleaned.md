
# Installation instructions
```
conda create --name nvsmask3d -y python=3.8
conda activate nvsmask3d
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# install sam
git clone --recurse-submodules git@github.com:junyuan-fang/nvsmask3d.git
cd  segment-anything/ #rememgber to download checkpoints to checkpoints/
pip install .
cd ..

```

### Before commiting your code. Format your code:
```
pip install ruff
ruff format . # formats files according to PEP8 style guide
```

### Make your python repo actually usable.
Update dependencies  in the `pyproject.toml` file. This allows a user to just run `pip install -e .` in the terminal and it will dowload all needed packages and dependencies. Right now, the user has to debug imports and not found files when they run simple commands `python nvsmask3d/scripts/` because dependencies are not taken care of. This helps moving code from local, to csc, and sharing code with others. 


python  results/segmentation/scene.py
python -m semantic.eval.eval_instance semantic/configs/eval_instance.yml

# info
tran test split = 1
test_mode = train

for scannetpp:

path of 3d mask proposals: ```/home/fangj1/Code/nerfstudio-nvsmask3d/nvsmask3d/data/mask3d_processed_first10``` corresponding scene's names are inside ```/home/fangj1/Code/nerfstudio-nvsmask3d/nvs_sub_val.txt```.

path of scannetpp data: ```/data/scannetpp/ScannetPP/data```

### SSH Port Forwarding (Tunneling)
``` 
ssh -L 7007:localhost:7007 <username>@<remote_machine_ip>
```

### train scannetpp scene
```
##dslr colmap (Use this, dump scenes to bash)
ns-train nvsmask3d --experiment-name 7b6477cb95_dslr_colmap --timestamp ""  --vis viewer scannetpp_nvsmask3d --data nvsmask3d/data/ScannetPP   --sequence 7b6477cb95 --mode dslr_colmap
```
### view scannetpp scene
```
ns-viewer nvsmask3d --load_config /home/fangj1/Code/nerfstudio-nvsmask3d/outputs/7b6477cb95/nvsmask3d/config.yml #iphone
ns-viewer nvsmask3d --load_config /home/fangj1/Code/nerfstudio-nvsmask3d/outputs/7b6477cb95_dslr_colmap/nvsmask3d/config.yml #dslr better pnsr. poses are from colmap

```

## train Replica scene
```
ns-train nvsmask3d --experiment-name office0 --timestamp "" --vis viewer replica_nvsmask3d --data nvsmask3d/data/replica --sequence office0 

```


## View
```
ns-viewer nvsmask3d --load_config outputs/office0/nvsmask3d/2024-08-14_204330/config.yml # office 0
ns-viewer nvsmask3d --load_config outputs/7b6477cb95/nvsmask3d/config.yml  #scannetpp

```

## Evaluation 
```

ns-eval for_ap --load_config nvsmask3d/data/replica


```
# Run evaluation in script
```
python nvsmask3d/script/eval_all.py
python /home/fangj1/Code/nerfstudio-nvsmask3d/nvsmask3d/script/eval_config.py #please use below mentioned code for SAM False replica version

    run_experiments(
        experiment_type="rgb",
        dataset="replica",
        sam=False,
        algorithm=0,
        project_name="crop",
        wandb_mode="disabled",
        kind="crop",
        interp_kind = "masked_gaussian", #average
    )
```
### for replica validation
```
ns-eval for_ap --load_config nvsmask3d/data/replica
```
## NVS quality eval
```
ns-eval psnr --load_config outputs/unnamed/nvsmask3d/2024-08-11_124613/config.yml
ns-eval psnr --load_config outputs/7b6477cb95_dslr_colmap/nvsmask3d/config.yml   # colmap eval

```
## Single scene predction path
```
/home/wangs9/junyuan/openmask3d/output/2024-07-23-11-44-44-scene0000_00_/scene0000_00__masks.pt
```


# Datasets
### Replica
Download dataset with: `python nvsmask3d/data/download_replica.py`
