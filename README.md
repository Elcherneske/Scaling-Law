# Scaling-Law.

## 论文结果复现

1. MultiPowerLaw

进入 MultiPowerLaw 文件夹，运行 `python main.py -f 100` 即可。

相关原始的 loss curve 在 `MultiPowerLaw/loss_curve_repo/csv_100` 文件夹中，名字分别为3stage_34000.csv，cosine_34000.csv，wsd_27000_34000.csv。

相关学习率调整参数手动设置在 `MultiPowerLaw/src/data_loader.py` 文件中调整。

2. Scaling_Law

进入 Scaling_Law 文件夹，运行 `python main.py` 即可， loss curve 在 `Scaling_Law/data` 文件夹中。

默认用cosine拟合并在wsd预测，如果需要调整step以及拟合和预测的loss函数，调整main.py中相关参数即可。

3. 基于warmup的初始化
使用warmup的结果见warmup_initial.zip文件中的test_with/without_I.ipynb，分别对应使用/不使用initial damping。

