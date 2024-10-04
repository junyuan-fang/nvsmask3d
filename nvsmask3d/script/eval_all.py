# run.py

from nvsmask3d.script.eval_config  import get_rgb_experiment, get_gaussian_experiment,get_mix_experiment
from tqdm import tqdm

experiments = get_rgb_experiment()#get_gaussian_experiment()#get_rgb_experiment()
for experiment in tqdm(experiments):
    experiment.run()

experiments = get_gaussian_experiment()#get_gaussian_experiment()#get_rgb_experiment()
for experiment in tqdm(experiments):
    experiment.run()

experiments = get_mix_experiment()#get_gaussian_experiment()#get_rgb_experiment()
for experiment in tqdm(experiments):
    experiment.run()

# test = experiments[0]
# print(test.run_name_for_wandb)
# test.run()
