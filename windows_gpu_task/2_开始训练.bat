@echo off
chcp 65001 >nul
title Drone-Ranger-RL 全部正式训练任务 (2个实验，共约3小时)
cd /d "%~dp0..\dqn_2d_standalone"

if not exist .venv (
    echo [错误] 还没安装环境，请先双击运行 1_安装环境.bat
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat

if not exist checkpoints\nav100_3obs_init.pt (
    echo [错误] 缺少起点模型 checkpoints\nav100_3obs_init.pt
    echo 请把对方发你的 checkpoints.zip 解压到 dqn_2d_standalone 文件夹
    pause
    exit /b 1
)

echo ============================================
echo  任务 1/2：living_cost=0.02，3000 轮
echo  期间日志滚动属正常，想中断可按 Ctrl+C
echo ============================================
python train_dqn.py --num_envs=64 --num_episodes=3000 --num_obstacles=3 --replay_buffer_size=5000000 --resume_from=checkpoints/nav100_3obs_init.pt --run_name=curr_3obs_lc --living_cost=0.02
if errorlevel 1 goto TRAIN_FAIL
echo 任务 1 训练完成，自动评估中...
python dqn_diagnose_2d_classify.py --checkpoint=checkpoints/curr_3obs_lc_final.pt --num_obstacles=3 --num_envs=64 --num_episodes=200 --csv_out=eval_lc002_final.csv

echo.
echo ============================================
echo  任务 2/2：living_cost=0.05，3000 轮
echo ============================================
python train_dqn.py --num_envs=64 --num_episodes=3000 --num_obstacles=3 --replay_buffer_size=5000000 --resume_from=checkpoints/nav100_3obs_init.pt --run_name=curr_3obs_lc05 --living_cost=0.05
if errorlevel 1 goto TRAIN_FAIL
echo 任务 2 训练完成，自动评估中...
python dqn_diagnose_2d_classify.py --checkpoint=checkpoints/curr_3obs_lc05_final.pt --num_obstacles=3 --num_envs=64 --num_episodes=200 --csv_out=eval_lc005_final.csv

echo.
echo ============================================
echo  全部任务完成！请把以下 4 个文件发回给项目方：
echo    1. dqn_2d_standalone\eval_lc002_final.csv
echo    2. dqn_2d_standalone\eval_lc005_final.csv
echo    3. dqn_2d_standalone\checkpoints\curr_3obs_lc_final.pt
echo    4. dqn_2d_standalone\checkpoints\curr_3obs_lc05_final.pt
echo ============================================
pause
exit /b 0

:TRAIN_FAIL
echo.
echo [错误] 训练异常退出，请截图发回给项目方
pause
exit /b 1
