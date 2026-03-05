with open("scripts/evaluate_metrics.py", "r") as f:
    text = f.read()

import re

# Remove imports
text = re.sub(r'compute_ciede2000,\s*', '', text)
text = re.sub(r'compute_ab_mse\s*', '', text)

# Remove metrics vars
text = re.sub(r'ciede\s*=\s*compute_ciede2000\(gt_img, pred_img\)', '', text)
text = re.sub(r'"CIEDE2000": ciede,', '', text)
text = re.sub(r'print\(f"CIEDE2000: \{summary\[\'CIEDE2000\'\]:\.4f\}"\)', '', text)
text = re.sub(r'f\.write\(f"CIEDE2000: \{summary\[\'CIEDE2000\'\]:\.4f\}\\n"\)', '', text)

# cleanup any weird commas in imports or dicts
with open("scripts/evaluate_metrics.py", "w") as f:
    f.write(text)
