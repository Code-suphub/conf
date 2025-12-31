from pathlib import Path
import numpy as np
from scipy.io import loadmat

def is_v5_compatible(path):
    try:
        data = loadmat(path, struct_as_record=False, squeeze_me=True)
    except Exception as e:
        return False, f"loadmat failed: {e}"

    total_bytes = 0
    for k, v in data.items():
        if k.startswith('__'):
            continue
        if not isinstance(v, np.ndarray):
            return False, f"{k} not ndarray"
        if v.dtype.kind == 'O':
            return False, f"{k} contains object"
        total_bytes += v.nbytes
        if total_bytes > 2**31 - 1:
            return False, "size > 2 GB"
    return True, "v5 compatible"

# 遍历目录下所有 .mat 文件
root = Path(r"matlab")          # 改成你的实际目录
for mat_file in root.rglob("*.mat"):
    ok, msg = is_v5_compatible(mat_file)
    print(f"{mat_file}: {ok} - {msg}")