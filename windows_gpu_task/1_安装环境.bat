@echo off
chcp 65001 >nul
title Drone-Ranger-RL 环境安装
cd /d "%~dp0..\dqn_2d_standalone"

echo ============================================
echo  正在创建虚拟环境 .venv ...
echo ============================================
python -m venv .venv
if errorlevel 1 (
    echo.
    echo [错误] 找不到 python，请确认安装时勾选了 "Add python.exe to PATH"
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo.
echo ============================================
echo  正在安装依赖（torch 较大，需要几分钟）...
echo ============================================
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [错误] 依赖安装失败，请检查网络后重试
    pause
    exit /b 1
)

echo.
echo ============================================
echo  安装完成！接下来请双击 2_开始训练.bat
echo ============================================
pause
