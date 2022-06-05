# SDIoU(Stadardized Distance-based IoU)
# Written by seareale
# https://github.com/seareale

import math

import torch


# code from YOLOv5
# https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/metrics.py#L216
def bbox_iou(
    box1,
    box2,
    xywh=True,
    GIoU=False,
    DIoU=False,
    CIoU=False,
    eps=1e-7,
    SDIoU=False,
    std_type=None,
):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, 1), box2.chunk(4, 1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if CIoU or DIoU or GIoU or SDIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU or SDIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            #####################################################################
            if std_type:  # SDIoU
                dist = sdiou((x1, y1, w1, h1), (x2, y2, w2, h2), std_type=std_type, eps=1e-7)
            #####################################################################
            else:  # DIoU
                c2 = cw**2 + ch**2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
                dist = rho2 / c2  # DIoU
            if (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (dist + v * alpha)  # CIoU
            return iou - dist  # DIoU or SDIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def sdiou(box1, box2, std_type="mean", eps=1e-7):
    if std_type == "mean":
        ##### mean distance
        w_mean = (box1[2] + box2[2]) / 2
        h_mean = (box1[3] + box2[3]) / 2

        dist = (box2[0] - box1[0]) ** 2 / (w_mean**2 + eps) + (box2[1] - box1[1]) ** 2 / (
            h_mean**2 + eps
        )
    elif std_type == "var":
        ##### variance distance
        w_var = (box1[2] ** 2 + box2[2] ** 2) / 2
        h_var = (box1[3] ** 2 + box2[3] ** 2) / 2

        dist = (box2[0] - box1[0]) ** 2 / (w_var + eps) + (box2[1] - box1[1]) ** 2 / (h_var + eps)
    else:
        raise NameError("No such std_type")

    return dist
