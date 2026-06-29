#!/bin/bash

git pull
source venv312/bin/activate
echo "正在安装依赖..."
pip install -r requirements.txt
echo "正在关闭旧的交易进程..."
pkill -f "examples/live_trading.py"
sleep 2 # 等待2秒确保完全释放端口和内存
echo "正在启动新的交易进程..."
nohup python examples/live_trading.py --log-level DEBUG > output.log 2>&1 &
echo "启动成功！当前最新的进程信息如下："
ps -ef | grep "examples/live_trading.py" | grep -v grep