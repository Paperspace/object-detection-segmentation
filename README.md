
## Paperspace - PyTorch-based modular object detection based on Detectron Demo

### Training & Evaluation

Please check out [docs on using Experiments with Paperspace](https://docs.paperspace.com/gradient/experiments/using-experiments)


We provide an example script in "training/train_net.py", that is made to train your model.
You may want to use it as a reference to write your own training script.

### Setup Dataset

This demo has builtin support for a few datasets.
Please check out [docs on using Datasets with Paperspace](https://docs.paperspace.com/gradient/experiments/using-experiments/experiment-datasets)

The datasets are assumed to exist in a directory `/data/DATASET`.
Under this directory, the script will look for datasets in the structure described below, if needed.
```
/data/coco/
```
```
# Example Code 
dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "/data"), "coco")
```
## Expected dataset structure for COCO instance/keypoint detection:

```
coco/
  annotations/
    instances_{train,val}2017.json
    person_keypoints_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

You can use the 2014 version of the dataset as well.

You can download a tiny version of the COCO dataset, with `./prepare_for_tests.sh`.
then run:
```
gradient experiments run singlenode \
  --name mnist \
  --projectId <your-project-id> \
  --container devopsbay/detectron2-cuda:v0 \
  --machineType p2.xlarge \
  --command "python training/train_net.py --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 --config-file ./configs/mask_rcnn_R_50_FPN_1x.yaml" \
  --workspaceUrl https://github.com/Paperspace/object-detection-segmentation
```