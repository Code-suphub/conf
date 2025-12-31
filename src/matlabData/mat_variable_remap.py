# rename_and_backup.py
import re
import numpy as np
from pathlib import Path
from scipy.io import loadmat, savemat

# ---------- 工具 ----------
def sanitize(name: str) -> str:
    """只保留字母、数字、下划线，且不以数字开头"""
    return re.sub(r'[^A-Za-z0-9_]', '_', name)

def rename_vars(data: dict):
    """返回 (新数据字典, 原→新映射字典)"""
    mapping = {}
    new_data = {}
    for old_key, val in data.items():
        if old_key.startswith('__'):           # 保留元数据
            new_data[old_key] = val
            continue
        new_key = sanitize(old_key)
        new_data[new_key] = val
        if new_key != old_key:
            mapping[old_key] = new_key
    return new_data, mapping

# ---------- 主循环 ----------
root = Path("matlab").parent      # 脚本所在目录；可改成 Path("matlab")
for mat_path in root.rglob("*.mat"):
    try:
        orig = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    except Exception as e:
        print(f"⚠️ 跳过 {mat_path}: {e}")
        continue

    new_data, rename_map = rename_vars(orig)

    # 打印映射
    if rename_map:
        print(f"\n📄 {mat_path.name}")
        for old, new in rename_map.items():
            print(f"   {old!r}  ->  {new!r}")

    # 写回（覆盖）
    try:
        savemat(mat_path, new_data, format='5', do_compression=True)
    except Exception as e:
        pass
print("✅ 全部完成！")



