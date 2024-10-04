import nvsmask3d
print(nvsmask3d.__file__)

from nvsmask3d.script.nvsmask3d_eval import ComputeForAP  # 导入你的 ComputeForAP 类
compute_ap = ComputeForAP(run_name_for_wandb="test", algorithm=2)

compute_ap.main()  