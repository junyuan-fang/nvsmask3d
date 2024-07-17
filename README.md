## Registering with Nerfstudio
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html). Clone or fork this repository and run the commands:

```
conda activate nerfstudio
cd nerfstudio-method-template/
pip install -e .
ns-install-cli
```

## Running the new method
This repository creates a new Nerfstudio method named "method-template". To train with it, run the command:
```
ns-train method-template --data [PATH]
ns-train splatfacto scannet-data --data nvsmask3d/data/scene0000_00_ 
ns-train nvsmask3d --data nvsmask3d/data/scene_example
ns-train nvsmask3d --data nvsmask3d/data/scene0000_00_ --vis viewer+wandb

```