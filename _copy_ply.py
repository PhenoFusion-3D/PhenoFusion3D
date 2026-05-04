import shutil, os
base = r"c:\Personal\My_DEGREE's\Master_of_Computing_(Advanced)\Australian_National_University\Technical Team Project - COMP8715"
src = os.path.join(base, "experiment", "data", "main", "test_plant_rs13_1", "output", "merge_pcd_best.ply")
dst_dir = os.path.join(base, "PhenoFusion3D", "sample_output")
os.makedirs(dst_dir, exist_ok=True)
if os.path.isfile(src):
    shutil.copy2(src, dst_dir)
    print("Copied:", src, "->", dst_dir)
else:
    print("SOURCE NOT FOUND:", src)
    print("Listing output dir:")
    out_dir = os.path.join(base, "experiment", "data", "main", "test_plant_rs13_1", "output")
    if os.path.isdir(out_dir):
        for f in os.listdir(out_dir):
            print(" ", f)
    else:
        print("  output dir not found:", out_dir)
