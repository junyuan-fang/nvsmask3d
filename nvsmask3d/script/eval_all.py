# run.py

from nvsmask3d.script.eval_config  import get_rgb_experiment, get_gaussian_experiment,get_mix_experiment
from tqdm import tqdm

# 获取实验配置
experiments = get_rgb_experiment()#get_gaussian_experiment()#get_rgb_experiment()

# #批量运行实验
# for experiment in tqdm(experiments):
#     experiment.run()
test = experiments[6]
print(test.run_name_for_wandb)
test.run()
