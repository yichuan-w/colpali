#!/bin/bash
# 复现60%结果的脚本

echo "===  创建Python 3.10虚拟环境 ==="
python3.10 -m venv venv_mteb_v1

echo "=== 激活环境 ==="
source venv_mteb_v1/bin/activate

echo "=== 回退MTEB到v1.39.7 ==="
cd mteb
git checkout d2c704c1
cd ..

echo "=== 安装MTEB v1.39.7 ==="
pip install -e ./mteb

echo "=== 验证MTEB版本 ==="
python3 -c "import mteb; print(f'MTEB版本: {mteb.__version__}')"

echo "=== 运行评估 ==="
python3 benchmark/vidorev2_all_bench.py

echo "=== 完成！检查results目录 ==="
