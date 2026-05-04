# sample_output

This directory holds reference reconstruction outputs for comparison.

## merge_pcd_best.ply

The best point cloud reconstruction obtained from the experiment pipeline
(`experiment/data/main/test_plant_rs13_1/output/merge_pcd_best.ply`).

To copy it here, run from the PhenoFusion3D directory:

```
python _copy_ply.py
```

Or manually copy the file from:

    ..\experiment\data\main\test_plant_rs13_1\output\merge_pcd_best.ply

To view it once copied:

```
python view_ply.py sample_output/merge_pcd_best.ply
```
