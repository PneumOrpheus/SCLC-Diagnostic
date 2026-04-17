RetinaNet Network
Part of this script is adapted from pytorch/vision

class monai.apps.detection.networks.retinanet_network.RetinaNet(spatial_dims, num_classes, num_anchors, feature_extractor, size_divisible=1, use_list_output=False)[source]
The network used in RetinaNet.

It takes an image tensor as inputs, and outputs either 1) a dictionary head_outputs. head_outputs[self.cls_key] is the predicted classification maps, a list of Tensor. head_outputs[self.box_reg_key] is the predicted box regression maps, a list of Tensor. or 2) a list of 2N tensors head_outputs, with first N tensors being the predicted classification maps and second N tensors being the predicted box regression maps.

Parameters
:
spatial_dims – number of spatial dimensions of the images. We support both 2D and 3D images.

num_classes – number of output classes of the model (excluding the background).

num_anchors – number of anchors at each location.

feature_extractor – a network that outputs feature maps from the input images, each feature map corresponds to a different resolution. Its output can have a format of Tensor, Dict[Any, Tensor], or Sequence[Tensor]. It can be the output of resnet_fpn_feature_extractor(*args, **kwargs).

size_divisible – the spatial size of the network input should be divisible by size_divisible, decided by the feature_extractor.

use_list_output – default False. If False, the network outputs a dictionary head_outputs, head_outputs[self.cls_key] is the predicted classification maps, a list of Tensor. head_outputs[self.box_reg_key] is the predicted box regression maps, a list of Tensor. If True, the network outputs a list of 2N tensors head_outputs, with first N tensors being the predicted classification maps and second N tensors being the predicted box regression maps.

Example

from monai.networks.nets import resnet
spatial_dims = 3  # 3D network
conv1_t_stride = (2,2,1)  # stride of first convolutional layer in backbone
backbone = resnet.ResNet(
    spatial_dims = spatial_dims,
    block = resnet.ResNetBottleneck,
    layers = [3, 4, 6, 3],
    block_inplanes = resnet.get_inplanes(),
    n_input_channels= 1,
    conv1_t_stride = conv1_t_stride,
    conv1_t_size = (7,7,7),
)
# This feature_extractor outputs 4-level feature maps.
# number of output feature maps is len(returned_layers)+1
returned_layers = [1,2,3]  # returned layer from feature pyramid network
feature_extractor = resnet_fpn_feature_extractor(
    backbone = backbone,
    spatial_dims = spatial_dims,
    pretrained_backbone = False,
    trainable_backbone_layers = None,
    returned_layers = returned_layers,
)
# This feature_extractor requires input image spatial size
# to be divisible by (32, 32, 16).
size_divisible = tuple(2*s*2**max(returned_layers) for s in conv1_t_stride)
model = RetinaNet(
    spatial_dims = spatial_dims,
    num_classes = 5,
    num_anchors = 6,
    feature_extractor=feature_extractor,
    size_divisible = size_divisible,
).to(device)
result = model(torch.rand(2, 1, 128,128,128))
cls_logits_maps = result["classification"]  # a list of len(returned_layers)+1 Tensor
box_regression_maps = result["box_regression"]  # a list of len(returned_layers)+1 Tensor
forward(images)[source]
It takes an image tensor as inputs, and outputs predicted classification maps and predicted box regression maps in head_outputs.

Parameters
:
images (Tensor) – input images, sized (B, img_channels, H, W) or (B, img_channels, H, W, D).

Return type
:
Any

Returns
:
1) If self.use_list_output is False, output a dictionary head_outputs with keys including self.cls_key and self.box_reg_key. head_outputs[self.cls_key] is the predicted classification maps, a list of Tensor. head_outputs[self.box_reg_key] is the predicted box regression maps, a list of Tensor. 2) if self.use_list_output is True, outputs a list of 2N tensors head_outputs, with first N tensors being the predicted classification maps and second N tensors being the predicted box regression maps.

class monai.apps.detection.networks.retinanet_network.RetinaNetClassificationHead(in_channels, num_anchors, num_classes, spatial_dims, prior_probability=0.01)[source]
A classification head for use in RetinaNet.

This head takes a list of feature maps as inputs, and outputs a list of classification maps. Each output map has same spatial size with the corresponding input feature map, and the number of output channel is num_anchors * num_classes.

Parameters
:
in_channels (int) – number of channels of the input feature

num_anchors (int) – number of anchors to be predicted

num_classes (int) – number of classes to be predicted

spatial_dims (int) – spatial dimension of the network, should be 2 or 3.

prior_probability (float) – prior probability to initialize classification convolutional layers.

forward(x)[source]
It takes a list of feature maps as inputs, and outputs a list of classification maps. Each output classification map has same spatial size with the corresponding input feature map, and the number of output channel is num_anchors * num_classes.

Parameters
:
x (list[Tensor]) – list of feature map, x[i] is a (B, in_channels, H_i, W_i) or (B, in_channels, H_i, W_i, D_i) Tensor.

Return type
:
list[Tensor]

Returns
:
cls_logits_maps, list of classification map. cls_logits_maps[i] is a (B, num_anchors * num_classes, H_i, W_i) or (B, num_anchors * num_classes, H_i, W_i, D_i) Tensor.

class monai.apps.detection.networks.retinanet_network.RetinaNetRegressionHead(in_channels, num_anchors, spatial_dims)[source]
A regression head for use in RetinaNet.

This head takes a list of feature maps as inputs, and outputs a list of box regression maps. Each output box regression map has same spatial size with the corresponding input feature map, and the number of output channel is num_anchors * 2 * spatial_dims.

Parameters
:
in_channels (int) – number of channels of the input feature

num_anchors (int) – number of anchors to be predicted

spatial_dims (int) – spatial dimension of the network, should be 2 or 3.

forward(x)[source]
It takes a list of feature maps as inputs, and outputs a list of box regression maps. Each output box regression map has same spatial size with the corresponding input feature map, and the number of output channel is num_anchors * 2 * spatial_dims.

Parameters
:
x (list[Tensor]) – list of feature map, x[i] is a (B, in_channels, H_i, W_i) or (B, in_channels, H_i, W_i, D_i) Tensor.

Return type
:
list[Tensor]

Returns
:
box_regression_maps, list of box regression map. cls_logits_maps[i] is a (B, num_anchors * 2 * spatial_dims, H_i, W_i) or (B, num_anchors * 2 * spatial_dims, H_i, W_i, D_i) Tensor.

monai.apps.detection.networks.retinanet_network.resnet_fpn_feature_extractor(backbone, spatial_dims, pretrained_backbone=False, returned_layers=(1, 2, 3), trainable_backbone_layers=None)[source]
Constructs a feature extractor network with a ResNet-FPN backbone, used as feature_extractor in RetinaNet.

Reference: “Focal Loss for Dense Object Detection”.

The returned feature_extractor network takes an image tensor as inputs, and outputs a dictionary that maps string to the extracted feature maps (Tensor).

The input to the returned feature_extractor is expected to be a list of tensors, each of shape [C, H, W] or [C, H, W, D], one for each image. Different images can have different sizes.

Parameters
:
backbone – a ResNet model, used as backbone.

spatial_dims – number of spatial dimensions of the images. We support both 2D and 3D images.

pretrained_backbone – whether the backbone has been pre-trained.

returned_layers – returned layers to extract feature maps. Each returned layer should be in the range [1,4]. len(returned_layers)+1 will be the number of extracted feature maps. There is an extra maxpooling layer LastLevelMaxPool() appended.

trainable_backbone_layers – number of trainable (not frozen) resnet layers starting from final block. Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable. When pretrained_backbone is False, this value is set to be 5. When pretrained_backbone is True, if None is passed (the default) this value is set to 3.

Example

from monai.networks.nets import resnet
spatial_dims = 3 # 3D network
backbone = resnet.ResNet(
    spatial_dims = spatial_dims,
    block = resnet.ResNetBottleneck,
    layers = [3, 4, 6, 3],
    block_inplanes = resnet.get_inplanes(),
    n_input_channels= 1,
    conv1_t_stride = (2,2,1),
    conv1_t_size = (7,7,7),
)
# This feature_extractor outputs 4-level feature maps.
# number of output feature maps is len(returned_layers)+1
feature_extractor = resnet_fpn_feature_extractor(
    backbone = backbone,
    spatial_dims = spatial_dims,
    pretrained_backbone = False,
    trainable_backbone_layers = None,
    returned_layers = [1,2,3],
)
model = RetinaNet(
    spatial_dims = spatial_dims,
    num_classes = 5,
    num_anchors = 6,
    feature_extractor=feature_extractor,
    size_divisible = 32,
).to(device)
RetinaNet Detector
Part of this script is adapted from pytorch/vision

class monai.apps.detection.networks.retinanet_detector.RetinaNetDetector(network, anchor_generator, box_overlap_metric=<function box_iou>, spatial_dims=None, num_classes=None, size_divisible=1, cls_key='classification', box_reg_key='box_regression', debug=False)[source]
Retinanet detector, expandable to other one stage anchor based box detectors in the future. An example of construction can found in the source code of retinanet_resnet50_fpn_detector() .

The input to the model is expected to be a list of tensors, each of shape (C, H, W) or (C, H, W, D), one for each image, and should be in 0-1 range. Different images can have different sizes. Or it can also be a Tensor sized (B, C, H, W) or (B, C, H, W, D). In this case, all images have same size.

The behavior of the model changes depending if it is in training or evaluation mode.

During training, the model expects both the input tensors, as well as a targets (list of dictionary), containing:

boxes (FloatTensor[N, 4] or FloatTensor[N, 6]): the ground-truth boxes in StandardMode, i.e., [xmin, ymin, xmax, ymax] or [xmin, ymin, zmin, xmax, ymax, zmax] format, with 0 <= xmin < xmax <= H, 0 <= ymin < ymax <= W, 0 <= zmin < zmax <= D.

labels: the class label for each ground-truth box

The model returns a Dict[str, Tensor] during training, containing the classification and regression losses. When saving the model, only self.network contains trainable parameters and needs to be saved.

During inference, the model requires only the input tensors, and returns the post-processed predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as follows:

boxes (FloatTensor[N, 4] or FloatTensor[N, 6]): the predicted boxes in StandardMode, i.e., [xmin, ymin, xmax, ymax] or [xmin, ymin, zmin, xmax, ymax, zmax] format, with 0 <= xmin < xmax <= H, 0 <= ymin < ymax <= W, 0 <= zmin < zmax <= D.

labels (Int64Tensor[N]): the predicted labels for each image

labels_scores (Tensor[N]): the scores for each prediction

Parameters
:
network – a network that takes an image Tensor sized (B, C, H, W) or (B, C, H, W, D) as input and outputs a dictionary Dict[str, List[Tensor]] or Dict[str, Tensor].

anchor_generator – anchor generator.

box_overlap_metric – func that compute overlap between two sets of boxes, default is Intersection over Union (IoU).

debug – whether to print out internal parameters, used for debugging and parameter tuning.

Notes

Input argument network can be a monai.apps.detection.networks.retinanet_network.RetinaNet(*) object, but any network that meets the following rules is a valid input network.

It should have attributes including spatial_dims, num_classes, cls_key, box_reg_key, num_anchors, size_divisible.

spatial_dims (int) is the spatial dimension of the network, we support both 2D and 3D.

num_classes (int) is the number of classes, excluding the background.

size_divisible (int or Sequence[int]) is the expectation on the input image shape. The network needs the input spatial_size to be divisible by size_divisible, length should be 2 or 3.

cls_key (str) is the key to represent classification in the output dict.

box_reg_key (str) is the key to represent box regression in the output dict.

num_anchors (int) is the number of anchor shapes at each location. it should equal to self.anchor_generator.num_anchors_per_location()[0].

If network does not have these attributes, user needs to provide them for the detector.

Its input should be an image Tensor sized (B, C, H, W) or (B, C, H, W, D).

About its output head_outputs, it should be either a list of tensors or a dictionary of str: List[Tensor]:

If it is a dictionary, it needs to have at least two keys: network.cls_key and network.box_reg_key, representing predicted classification maps and box regression maps. head_outputs[network.cls_key] should be List[Tensor] or Tensor. Each Tensor represents classification logits map at one resolution level, sized (B, num_classes*num_anchors, H_i, W_i) or (B, num_classes*num_anchors, H_i, W_i, D_i). head_outputs[network.box_reg_key] should be List[Tensor] or Tensor. Each Tensor represents box regression map at one resolution level, sized (B, 2*spatial_dims*num_anchors, H_i, W_i)or (B, 2*spatial_dims*num_anchors, H_i, W_i, D_i). len(head_outputs[network.cls_key]) == len(head_outputs[network.box_reg_key]).

If it is a list of 2N tensors, the first N tensors should be the predicted classification maps, and the second N tensors should be the predicted box regression maps.

Example

# define a naive network
import torch
class NaiveNet(torch.nn.Module):
    def __init__(self, spatial_dims: int, num_classes: int):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.num_classes = num_classes
        self.size_divisible = 2
        self.cls_key = "cls"
        self.box_reg_key = "box_reg"
        self.num_anchors = 1
    def forward(self, images: torch.Tensor):
        spatial_size = images.shape[-self.spatial_dims:]
        out_spatial_size = tuple(s//self.size_divisible for s in spatial_size)  # half size of input
        out_cls_shape = (images.shape[0],self.num_classes*self.num_anchors) + out_spatial_size
        out_box_reg_shape = (images.shape[0],2*self.spatial_dims*self.num_anchors) + out_spatial_size
        return {self.cls_key: [torch.randn(out_cls_shape)], self.box_reg_key: [torch.randn(out_box_reg_shape)]}

# create a RetinaNetDetector detector
spatial_dims = 3
num_classes = 5
anchor_generator = monai.apps.detection.utils.anchor_utils.AnchorGeneratorWithAnchorShape(
    feature_map_scales=(1, ), base_anchor_shapes=((8,) * spatial_dims)
)
net = NaiveNet(spatial_dims, num_classes)
detector = RetinaNetDetector(net, anchor_generator)

# only detector.network may contain trainable parameters.
optimizer = torch.optim.SGD(
    detector.network.parameters(),
    1e-3,
    momentum=0.9,
    weight_decay=3e-5,
    nesterov=True,
)
torch.save(detector.network.state_dict(), 'model.pt')  # save model
detector.network.load_state_dict(torch.load('model.pt', weights_only=True))  # load model
compute_anchor_matched_idxs(anchors, targets, num_anchor_locs_per_level)[source]
Compute the matched indices between anchors and ground truth (gt) boxes in targets. output[k][i] represents the matched gt index for anchor[i] in image k. Suppose there are M gt boxes for image k. The range of it output[k][i] value is [-2, -1, 0, …, M-1]. [0, M - 1] indicates this anchor is matched with a gt box, while a negative value indicating that it is not matched.

Parameters
:
anchors (list[Tensor]) – a list of Tensor. Each Tensor represents anchors for each image, sized (sum(HWA), 2*spatial_dims) or (sum(HWDA), 2*spatial_dims). A = self.num_anchors_per_loc.

targets (list[dict[str, Tensor]]) – a list of dict. Each dict with two keys: self.target_box_key and self.target_label_key, ground-truth boxes present in the image.

num_anchor_locs_per_level (Sequence[int]) – each element represents HW or HWD at this level.

Return type
:
list[Tensor]

Returns
:
a list of matched index matched_idxs_per_image (Tensor[int64]), Tensor sized (sum(HWA),) or (sum(HWDA),). Suppose there are M gt boxes. matched_idxs_per_image[i] is a matched gt index in [0, M - 1] or a negative value indicating that anchor i could not be matched. BELOW_LOW_THRESHOLD = -1, BETWEEN_THRESHOLDS = -2

compute_box_loss(box_regression, targets, anchors, matched_idxs)[source]
Compute box regression losses.

Parameters
:
box_regression (Tensor) – box regression results, sized (B, sum(HWA), 2*self.spatial_dims)

targets (list[dict[str, Tensor]]) – a list of dict. Each dict with two keys: self.target_box_key and self.target_label_key, ground-truth boxes present in the image.

anchors (list[Tensor]) – a list of Tensor. Each Tensor represents anchors for each image, sized (sum(HWA), 2*spatial_dims) or (sum(HWDA), 2*spatial_dims). A = self.num_anchors_per_loc.

matched_idxs (list[Tensor]) – a list of matched index. each element is sized (sum(HWA),) or (sum(HWDA),)

Return type
:
Tensor

Returns
:
box regression losses.

compute_cls_loss(cls_logits, targets, matched_idxs)[source]
Compute classification losses.

Parameters
:
cls_logits (Tensor) – classification logits, sized (B, sum(HW(D)A), self.num_classes)

targets (list[dict[str, Tensor]]) – a list of dict. Each dict with two keys: self.target_box_key and self.target_label_key, ground-truth boxes present in the image.

matched_idxs (list[Tensor]) – a list of matched index. each element is sized (sum(HWA),) or (sum(HWDA),)

Return type
:
Tensor

Returns
:
classification losses.

compute_loss(head_outputs_reshape, targets, anchors, num_anchor_locs_per_level)[source]
Compute losses.

Parameters
:
head_outputs_reshape (dict[str, Tensor]) – reshaped head_outputs. head_output_reshape[self.cls_key] is a Tensor sized (B, sum(HW(D)A), self.num_classes). head_output_reshape[self.box_reg_key] is a Tensor sized (B, sum(HW(D)A), 2*self.spatial_dims)

targets (list[dict[str, Tensor]]) – a list of dict. Each dict with two keys: self.target_box_key and self.target_label_key, ground-truth boxes present in the image.

anchors (list[Tensor]) – a list of Tensor. Each Tensor represents anchors for each image, sized (sum(HWA), 2*spatial_dims) or (sum(HWDA), 2*spatial_dims). A = self.num_anchors_per_loc.

Return type
:
dict[str, Tensor]

Returns
:
a dict of several kinds of losses.

forward(input_images, targets=None, use_inferer=False)[source]
Returns a dict of losses during training, or a list predicted dict of boxes and labels during inference.

Parameters
:
input_images – The input to the model is expected to be a list of tensors, each of shape (C, H, W) or (C, H, W, D), one for each image, and should be in 0-1 range. Different images can have different sizes. Or it can also be a Tensor sized (B, C, H, W) or (B, C, H, W, D). In this case, all images have same size.

targets – a list of dict. Each dict with two keys: self.target_box_key and self.target_label_key, ground-truth boxes present in the image (optional).

use_inferer – whether to use self.inferer, a sliding window inferer, to do the inference. If False, will simply forward the network. If True, will use self.inferer, and requires self.set_sliding_window_inferer(*args) to have been called before.

Returns
:
If training mode, will return a dict with at least two keys, including self.cls_key and self.box_reg_key, representing classification loss and box regression loss.

If evaluation mode, will return a list of detection results. Each element corresponds to an images in input_images, is a dict with at least three keys, including self.target_box_key, self.target_label_key, self.pred_score_key, representing predicted boxes, classification labels, and classification scores.

generate_anchors(images, head_outputs)[source]
Generate anchors and store it in self.anchors: List[Tensor]. We generate anchors only when there is no stored anchors, or the new coming images has different shape with self.previous_image_shape

Parameters
:
images (Tensor) – input images, a (B, C, H, W) or (B, C, H, W, D) Tensor.

head_outputs (dict[str, list[Tensor]]) – head_outputs. head_output_reshape[self.cls_key] is a Tensor sized (B, sum(HW(D)A), self.num_classes). head_output_reshape[self.box_reg_key] is a Tensor sized (B, sum(HW(D)A), 2*self.spatial_dims)

Return type
:
None

get_box_train_sample_per_image(box_regression_per_image, targets_per_image, anchors_per_image, matched_idxs_per_image)[source]
Get samples from one image for box regression losses computation.

Parameters
:
box_regression_per_image (Tensor) – box regression result for one image, (sum(HWA), 2*self.spatial_dims)

targets_per_image (dict[str, Tensor]) – a dict with at least two keys: self.target_box_key and self.target_label_key, ground-truth boxes present in the image.

anchors_per_image (Tensor) – anchors of one image, sized (sum(HWA), 2*spatial_dims) or (sum(HWDA), 2*spatial_dims). A = self.num_anchors_per_loc.

matched_idxs_per_image (Tensor) – matched index, sized (sum(HWA),) or (sum(HWDA),)

Return type
:
tuple[Tensor, Tensor]

Returns
:
paired predicted and GT samples from one image for box regression losses computation

get_cls_train_sample_per_image(cls_logits_per_image, targets_per_image, matched_idxs_per_image)[source]
Get samples from one image for classification losses computation.

Parameters
:
cls_logits_per_image (Tensor) – classification logits for one image, (sum(HWA), self.num_classes)

targets_per_image (dict[str, Tensor]) – a dict with at least two keys: self.target_box_key and self.target_label_key, ground-truth boxes present in the image.

matched_idxs_per_image (Tensor) – matched index, Tensor sized (sum(HWA),) or (sum(HWDA),) Suppose there are M gt boxes. matched_idxs_per_image[i] is a matched gt index in [0, M - 1] or a negative value indicating that anchor i could not be matched. BELOW_LOW_THRESHOLD = -1, BETWEEN_THRESHOLDS = -2

Return type
:
tuple[Tensor, Tensor]

Returns
:
paired predicted and GT samples from one image for classification losses computation

postprocess_detections(head_outputs_reshape, anchors, image_sizes, num_anchor_locs_per_level, need_sigmoid=True)[source]
Postprocessing to generate detection result from classification logits and box regression. Use self.box_selector to select the final output boxes for each image.

Parameters
:
head_outputs_reshape (dict[str, Tensor]) – reshaped head_outputs. head_output_reshape[self.cls_key] is a Tensor sized (B, sum(HW(D)A), self.num_classes). head_output_reshape[self.box_reg_key] is a Tensor sized (B, sum(HW(D)A), 2*self.spatial_dims)

targets – a list of dict. Each dict with two keys: self.target_box_key and self.target_label_key, ground-truth boxes present in the image.

anchors (list[Tensor]) – a list of Tensor. Each Tensor represents anchors for each image, sized (sum(HWA), 2*spatial_dims) or (sum(HWDA), 2*spatial_dims). A = self.num_anchors_per_loc.

Return type
:
list[dict[str, Tensor]]

Returns
:
a list of dict, each dict corresponds to detection result on image.

set_atss_matcher(num_candidates=4, center_in_gt=False)[source]
Using for training. Set ATSS matcher that matches anchors with ground truth boxes

Parameters
:
num_candidates (int) – number of positions to select candidates from. Smaller value will result in a higher matcher threshold and less matched candidates.

center_in_gt (bool) – If False (default), matched anchor center points do not need to lie within the ground truth box. Recommend False for small objects. If True, will result in a strict matcher and less matched candidates.

Return type
:
None

set_balanced_sampler(batch_size_per_image, positive_fraction)[source]
Using for training. Set torchvision balanced sampler that samples part of the anchors for training.

Parameters
:
batch_size_per_image (int) – number of elements to be selected per image

positive_fraction (float) – percentage of positive elements per batch

Return type
:
None

set_box_coder_weights(weights)[source]
Set the weights for box coder.

Parameters
:
weights (tuple[float]) – a list/tuple with length of 2*self.spatial_dims

Return type
:
None

set_box_regression_loss(box_loss, encode_gt, decode_pred)[source]
Using for training. Set loss for box regression.

Parameters
:
box_loss (Module) – loss module for box regression

encode_gt (bool) – if True, will encode ground truth boxes to target box regression before computing the losses. Should be True for L1 loss and False for GIoU loss.

decode_pred (bool) – if True, will decode predicted box regression into predicted boxes before computing losses. Should be False for L1 loss and True for GIoU loss.

Example

detector.set_box_regression_loss(
    torch.nn.SmoothL1Loss(beta=1.0 / 9, reduction="mean"),
    encode_gt = True, decode_pred = False
)
detector.set_box_regression_loss(
    monai.losses.giou_loss.BoxGIoULoss(reduction="mean"),
    encode_gt = False, decode_pred = True
)
Return type
:
None

set_box_selector_parameters(score_thresh=0.05, topk_candidates_per_level=1000, nms_thresh=0.5, detections_per_img=300, apply_sigmoid=True)[source]
Using for inference. Set the parameters that are used for box selection during inference. The box selection is performed with the following steps:

For each level, discard boxes with scores less than self.score_thresh.

For each level, keep boxes with top self.topk_candidates_per_level scores.

For the whole image, perform non-maximum suppression (NMS) on boxes, with overlapping threshold nms_thresh.

For the whole image, keep boxes with top self.detections_per_img scores.

Parameters
:
score_thresh (float) – no box with scores less than score_thresh will be kept

topk_candidates_per_level (int) – max number of boxes to keep for each level

nms_thresh (float) – box overlapping threshold for NMS

detections_per_img (int) – max number of boxes to keep for each image

Return type
:
None

set_cls_loss(cls_loss)[source]
Using for training. Set loss for classification that takes logits as inputs, make sure sigmoid/softmax is built in.

Parameters
:
cls_loss (Module) – loss module for classification

Example

detector.set_cls_loss(torch.nn.BCEWithLogitsLoss(reduction="mean"))
detector.set_cls_loss(FocalLoss(reduction="mean", gamma=2.0))
Return type
:
None

set_hard_negative_sampler(batch_size_per_image, positive_fraction, min_neg=1, pool_size=10)[source]
Using for training. Set hard negative sampler that samples part of the anchors for training.

HardNegativeSampler is used to suppress false positive rate in classification tasks. During training, it select negative samples with high prediction scores.

Parameters
:
batch_size_per_image (int) – number of elements to be selected per image

positive_fraction (float) – percentage of positive elements in the selected samples

min_neg (int) – minimum number of negative samples to select if possible.

pool_size (float) – when we need num_neg hard negative samples, they will be randomly selected from num_neg * pool_size negative samples with the highest prediction scores. Larger pool_size gives more randomness, yet selects negative samples that are less ‘hard’, i.e., negative samples with lower prediction scores.

Return type
:
None

set_regular_matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)[source]
Using for training. Set torchvision matcher that matches anchors with ground truth boxes.

Parameters
:
fg_iou_thresh (float) – foreground IoU threshold for Matcher, considered as matched if IoU > fg_iou_thresh

bg_iou_thresh (float) – background IoU threshold for Matcher, considered as not matched if IoU < bg_iou_thresh

allow_low_quality_matches (bool) – if True, produce additional matches for predictions that have only low-quality match candidates.

Return type
:
None

set_sliding_window_inferer(roi_size, sw_batch_size=1, overlap=0.5, mode=constant, sigma_scale=0.125, padding_mode=constant, cval=0.0, sw_device=None, device=None, progress=False, cache_roi_weight_map=False)[source]
Define sliding window inferer and store it to self.inferer.

set_target_keys(box_key, label_key)[source]
Set keys for the training targets and inference outputs. During training, both box_key and label_key should be keys in the targets when performing self.forward(input_images, targets). During inference, they will be the keys in the output dict of self.forward(input_images)`.

Return type
:
None

monai.apps.detection.networks.retinanet_detector.retinanet_resnet50_fpn_detector(num_classes, anchor_generator, returned_layers=(1, 2, 3), pretrained=False, progress=True, **kwargs)[source]
Returns a RetinaNet detector using a ResNet-50 as backbone, which can be pretrained from Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf> _.

Parameters
:
num_classes (int) – number of output classes of the model (excluding the background).

anchor_generator (AnchorGenerator) – AnchorGenerator,

returned_layers (Sequence[int]) – returned layers to extract feature maps. Each returned layer should be in the range [1,4]. len(returned_layers)+1 will be the number of extracted feature maps. There is an extra maxpooling layer LastLevelMaxPool() appended.

pretrained (bool) – If True, returns a backbone pre-trained on 23 medical datasets

progress (bool) – If True, displays a progress bar of the download to stderr

Return type
:
RetinaNetDetector

Returns
:
A RetinaNetDetector object with resnet50 as backbone

Example

# define a naive network
resnet_param = {
    "pretrained": False,
    "spatial_dims": 3,
    "n_input_channels": 2,
    "num_classes": 3,
    "conv1_t_size": 7,
    "conv1_t_stride": (2, 2, 2)
}
returned_layers = [1]
anchor_generator = monai.apps.detection.utils.anchor_utils.AnchorGeneratorWithAnchorShape(
    feature_map_scales=(1, 2), base_anchor_shapes=((8,) * resnet_param["spatial_dims"])
)
detector = retinanet_resnet50_fpn_detector(
    **resnet_param, anchor_generator=anchor_generator, returned_layers=returned_layers
)
Transforms
monai.apps.detection.transforms.box_ops.apply_affine_to_boxes(boxes, affine)[source]
