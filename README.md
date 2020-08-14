![GitHubSplash](https://user-images.githubusercontent.com/585865/65443342-e630d300-ddfb-11e9-9bcd-de1d2033ea60.png)

Paperspace - PyTorch-based modular object detection based on Detectron Demo
=================
<br>

**Get started:** [Create Account](https://www.paperspace.com/account/signup?gradient=true) • [Install CLI](https://docs.paperspace.com/gradient/get-started/install-the-cli) • [Tutorials](https://docs.paperspace.com/gradient/tutorials) • [Docs](https://docs.paperspace.com/gradient)

**Resources:** [Website](https://gradient.paperspace.com/) • [Blog](https://blog.paperspace.com/) • [Support](https://support.paperspace.com/hc/en-us) • [Contact Sales](https://use.paperspace.com/contact-sales)

<br>
=================

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

_Note: training on a single will take a long time, so be prepared to wait!_

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
The coco dataset is downloaded to the `./data/coco/traing2017` directory.
Model results are stored in the `./models` directory.

## How to deploy model on Gradient

This example will load previously trained model and launch a web app application with simple gui 

```
gradient deployments create \
  --name mask_rcnn4 --instanceCount 1 \
  --imageUrl devopsbay/detectron2-cuda:v0 \
  --machineType p2.xlarge \
  --command "sudo python demo/app.py" \
  --workspace https://github.com/Paperspace/object-detection-segmentation.git \               
  --deploymentType Custom \
  --clusterId <cluster id> \
  --modelId <model id>
```
![Example](demo/samples/detect.jpeg?raw=true "Example Object Detection")