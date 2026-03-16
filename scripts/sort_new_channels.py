import os, re, shutil

prepared = "/scratch/gilbreth/ddstephe/vivm/prepared"

for folder_tag, staging in [("angle_mp", "/scratch/gilbreth/ddstephe/vivm/angle_mp_staging"),
                              ("gradient", "/scratch/gilbreth/ddstephe/vivm/gradient_staging")]:
    files = os.listdir(staging)
    print(f"\n{folder_tag}: {len(files)} files")
    missing = []
for fname in files:
    # "approaching_leaving (1)_m.png" -> "approaching_leaving 1"
    stem = os.path.splitext(fname)[0]
    clip = re.sub(r' \((\d+)\)_m$', r' \1', stem)  # "approaching (1)_m" -> "approaching 1"
    if clip == stem:  # regex didn't match, just strip _m
        clip = re.sub(r'_m$', '', stem)              # "crossing_noV_front_m" -> "crossing_noV_front"
    clip_dir = os.path.join(prepared, clip)
    if not os.path.exists(clip_dir):
        missing.append(clip)
        continue
    dest_dir = os.path.join(clip_dir, folder_tag)
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(os.path.join(staging, fname), os.path.join(dest_dir, fname))
if missing:
    print(f"  WARNING - no clip dir found for: {missing}")
else:
    print(f"  All files placed successfully")