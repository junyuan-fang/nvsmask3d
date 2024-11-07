
python  results/segmentation/scene.py
python -m semantic.eval.eval_instance semantic/configs/eval_instance.yml
## env
salloc --job-name=scannetpp_eval --account=project_2003267 --partition=gpu --time=4:15:00 --ntasks=1 --cpus-per-task=6 --mem=30G --gres=gpu:v100:1,nvme:30

conda activate /scratch/project_2003267/miniforge/envs/nerfstudio

source /scratch/project_2003267/miniforge/bin/activate /scratch/project_2003267/miniforge/envs/nerfstudio
cd /scratch/project_2003267/nvsmask3d

python nvsmask3d/automation_test.py 

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
