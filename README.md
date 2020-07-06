# kitti_completion
Code for `Revisiting Sparsity Invariant Convolution: A Network for Image Guided Depth Completion`.

## KITTI training data
You need download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) to train.

**Warning:** it weighs about **175GB**, so make sure you have enough space to unzip too!

Our default settings expect that you have converted the png images to jpeg.

## Generate KITTI Submission
```
python dump_test.py --weight_path ./pretrained_weights.pth --data_path /path/to/dir/has/test_depth_completion_anonymous --dump
```
If you want to view results, please remove `--dump` option.

## Training
```
python train.py --data_path /path/to/kitti --split full
```
The path containing KITTI datasets should be organized like this:
```
/path/to/kitti
    raw/
        2011_09_26/
            2011_09_26_drive_0001_sync/
                image_02/
                image_03/
    test_depth_completion_anonymous/
        image/
        velodyne_raw/
    val_selection_cropped/
        image/
        velodyne_raw/
        groundtruth_depth/
```
