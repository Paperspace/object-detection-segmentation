
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
#### Expected dataset structure for COCO instance/keypoint detection:

```
coco/
  annotations/
    instances_{train,val}2017.json
    person_keypoints_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```
You can download a tiny version of the COCO dataset, with `training/download_coco.sh`.

#### COCO Dataset
Probably the most widely used dataset today for object localization is COCO: Common Objects in Context. Provided here are all the files from the 2017 version, along with an additional subset dataset created by fast.ai. Details of each COCO dataset is available from the COCO dataset page. The fast.ai subset contains all images that contain one of five selected categories, restricting objects to just those five categories; the categories are: chair couch tv remote book vase.

[fast.ai subset](https://s3.amazonaws.com/fast-ai-coco/coco_sample.tgz)

[Train images](https://s3.amazonaws.com/fast-ai-coco/train2017.zip)

## Run Training on Gradient

### Gradient CLI Installation

How to install Gradient CLI - [docs](https://docs.paperspace.com/gradient/get-started/install-the-cli)

```
pip install gradient --pre
```
Then make sure to [obtain an API Key](https://docs.paperspace.com/gradient/get-started/install-the-cli#obtaining-an-api-key), and then:
```
gradient apiKey XXXXXXXXXXXXXXXXXXX
```

### Train on a single GPU
```
gradient experiments run singlenode \
  --name mask_rcnn \
  --projectId <some project> \
  --container devopsbay/detectron2-cuda:v0 \
  --machineType p2.xlarge \
  --command "sudo python training/train_net.py --config-file training/configs/mask_rcnn_R_50_FPN_1x.yaml --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025" \
  --workspace https://github.com/Paperspace/object-detection-segmentation.git \
  --datasetName coco \
  --datasetUri s3://fast-ai-coco/train2017.zip \
  --clusterId <cluster id>
```

