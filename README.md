![GitHubSplash](https://user-images.githubusercontent.com/585865/65443342-e630d300-ddfb-11e9-9bcd-de1d2033ea60.png)

Gradient - PyTorch-based modular object detection based on Detectron Demo
=================
<br>

**Get started:** [Create Account](https://console.paperspace.com/signup?gradient=true) • [Install CLI](https://docs.paperspace.com/gradient/get-started/install-the-cli) • [Tutorials](https://docs.paperspace.com/gradient/tutorials) • [Docs](https://docs.paperspace.com/gradient)

**Resources:** [Website](https://gradient.paperspace.com/) • [Blog](https://blog.paperspace.com/) • [Support](https://support.paperspace.com/hc/en-us) • [Contact Sales](https://info.paperspace.com/contact-sales)

<br>
=================

### Blog Post

This Repository is related to our [Blog post](https://blog.paperspace.com/object-detection-segmentation-with-detectron2-on-paperspace-gradient/) 


### Training & Evaluation

Please check out [docs on using Experiments with Gradient](https://docs.paperspace.com/gradient/experiments/using-experiments)

We provide an example script in "training/train_net.py" that is made to train your model. 
You can use this as a reference to write your own training script.

### Setup Dataset

This demo has built-in support for a few datasets.
Please check out [docs on using Datasets with Gradient](https://docs.paperspace.com/gradient/experiments/using-experiments/experiment-datasets)

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
pip install gradient
```
Then make sure to [obtain an API Key](https://docs.paperspace.com/gradient/get-started/install-the-cli#obtaining-an-api-key), and then:
```
gradient apiKey XXXXXXXXXXXXXXXXXXX
```

### Train on a single GPU

_Note: training on a single will take a long time, so be prepared to wait!_

```
gradient experiments run singlenode \
  --name detectron2-demo \
  --projectId pr3qnl0g8 \
  --container devopsbay/detectron2:v1 \
  --machineType P4000 \
  --command "sudo python training/train_net.py --config-file training/configs/mask_rcnn_R_50_FPN_1x.yaml --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 MODEL.WEIGHTS https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_1x/137257794/model_final_b275ba.pkl  OUTPUT_DIR /artifacts/models/detectron" \
  --workspace https://github.com/Paperspace/object-detection-segmentation.git \
  --datasetName small_coco \
  --datasetUri s3://paperspace-tiny-coco/small_coco.zip \
  --clusterId <Cluster ID>
```
The coco dataset is downloaded to the `./data/coco/traing2017` directory.
Model results are stored in the `./models` directory.

### Running distributed training on a Gradient private cluster
In order to run a an experiment on a [Gradient private cluster](https://docs.paperspace.com/gradient/gradient-private-cloud/about), we need to add few additional parameters:
```
gradient experiments run multinode \
  --name mask_rcnn_multinode \
  --projectId <some project> \
  --workerContainer devopsbay/detectron2:v1 \
  --workerMachineType P4000 \
  --workerCount 7 \
  --parameterServerContainer devopsbay/detectron2:v1 \
  --parameterServerMachineType P4000 \
  --parameterServerCount 1 \
  --experimentType GRPC \
  --workerCommand "python training/train_net.py --config-file training/configs/mask_rcnn_R_50_FPN_1x.yaml --num-machines 8 MODEL.WEIGHTS https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_1x/137257794/model_final_b275ba.pkl  OUTPUT_DIR /artifacts/models/detectron" \
  --parameterServerCommand "ython training/train_net.py --config-file training/configs/mask_rcnn_R_50_FPN_1x.yaml --num-machines 8 MODEL.WEIGHTS https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_1x/137257794/model_final_b275ba.pkl  OUTPUT_DIR /artifacts/models/detectron" \
  --workspace https://github.com/Paperspace/object-detection-segmentation.git \
  --datasetName small_coco \
  --datasetUri s3://paperspace-tiny-coco/small_coco.zip \
  --clusterId <cluster id>
```

## How to deploy model on Gradient

This example will load previously trained model and launch a web app application with simple interface. 

```
deployments create /
--name paperspace-detectron-demo-app /
--instanceCount 1 /
--imageUrl devopsbay/detectron2:v1 /
--machineType V100 /
--command "pip3 install -r demo/requirements.txt && python demo/app.py" /
--workspace https://github.com/Paperspace/object-detection-segmentation.git 
--deploymentType Custom 
--clusterId <cluster id> 
--modelId <model id> 
--ports 8080
```
![Example](demo/samples/detect.jpeg?raw=true "Example Object Detection")

### Adding custom metrics to Inference app

Inside demo/ObjectDetector.py you will find a simple example to push custom metrics into gradient.
[Docs](https://docs.paperspace.com/gradient/metrics/push-metrics)

Example code
```
from gradient_utils.metrics import MetricsLogger

logger = MetricsLogger()
logger.add_counter("inference_count")
logger["inference_count"].inc()
logger.push_metrics()
```
