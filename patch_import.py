import sys

filepath = "d:/EyetechCode/finetune_dynamic_lora_prunning_snapflow.py"
with open(filepath, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the start of PHASE 3 mapping
start_idx = -1
for i, line in enumerate(lines):
    if "PHASE 3: Training Phase 1" in line:
        start_idx = i - 2
        break

if start_idx != -1 and "if __name__ == \"__main__\":" not in "".join(lines):
    new_lines = lines[:start_idx]
    new_lines.append("\nif __name__ == \"__main__\":\n")
    for line in lines[start_idx:]:
        if line.strip() == "":
            new_lines.append(line)
        else:
            new_lines.append("    " + line)
            
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print("Patched successfully")
else:
    print("Already patched or could not find PHASE 3")
