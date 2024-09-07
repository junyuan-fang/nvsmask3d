# run.py

from nvsmask3d.script.eval_config  import get_experiments
from tqdm import tqdm

# 获取实验配置
experiments = get_experiments()

# 批量运行实验
for experiment in tqdm(experiments):
    experiment.run()
