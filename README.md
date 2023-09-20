# SDIoU
This is an IoU metric that replaces DIoU using standardized distance.

<br/>

## <div align="center">Introduction</div>
There are two problems when using DIoU[[1]](https://arxiv.org/abs/1911.08287): Position, Scale. Those are caused by a normalization factor affected by the center point distance of bounding boxes. So we implemented Standardized Distance-based IoU metric using different normalization factors(SDIoU).

DIoU based on $\theta$ can be calculated as the following equation:  

$$
\begin{equation}
R_{DIoU} = \frac{(d\cdot cos\theta)^2 + (d\cdot sin\theta)^2}
{(\frac{w}{2} + \frac{w'}{2} + d\cdot cos\theta)^2 + (\frac{h}{2} + \frac{h'}{2} + d\cdot sin\theta)^2}
\end{equation}
$$

For easy understanding, the figure below shows each elements of the equation.

<div align="center">
<img src="https://wiki.seareale.dev/uploads/images/gallery/2022-06/Zks3A6Zf4DoNnVfP-image-1654425738734.png" width=40% hspace=20/>
<p>An illustration of DIoU calculated based on θ</p>
</div>

Following the equation based on $\theta$, the normalization factor is affected by the center point distance of bounding boxes. And the problems resulting from this can be seen in the figure below.

<div align="center">
  <img src="https://wiki.seareale.dev/uploads/images/gallery/2022-06/chXSKAk4rTl9Axj0-image-1654430531666.png" hspace=20/>
  <p>Illustrations of the position, scale problems of DIoU</p>
</div>

<br/>
<br/>

## <div align="center">Details</div>
The standardized distance[[2]](https://en.wikipedia.org/wiki/Distance) uses the variance of each axis as a normalization factor. And we replaced the variance value with the equation for the height and width of two bounding boxes as follows:
<div align="center">
<img src="https://wiki.seareale.dev/uploads/images/gallery/2022-06/rGNNfM9XMtLMrnI5-image-1654429713446.png" hspace=20 width=80%/>
    <p>Induction of the variance value of SDIoU</p>
</div>

The values of $SD_x$ and $SD_y$ get 0 ~ 4 when the two bounding boxes overlap. So, we divided them by 4 for normalization. The final equation of SDIoU can be calculated as shown in the following figure.


<div align="center">
<img src="https://wiki.seareale.dev/uploads/images/gallery/2022-06/Q2VvgTpVWdshXQRQ-image-1654427600274.png" hspace=20 width=50%/>
    <p>Comparison between DIoU and SDIoU</p>
</div>

And SDIoU(Standardized Distance-based IoU) solved all the problems of DIoU: Position, Scale.

<div align="center">
  <img src="https://wiki.seareale.dev/uploads/images/gallery/2022-06/5WTEuPfaNeOPJ1lX-image-1654426596689.png" hspace=20 width=80%/>
  <p>Illustrations of SDIoU that solves the problems of DIoU.</p>
</div>


<br/>
<br/>

## <div align="center">How to use</div>
1. run the command
```bash
$ pip install -r requirements.txt
```

2. initialize variables and add the code in your **Loss function** like below.
```python
from sidou import * 

  ...
  pred = ... # prediction bounding box ((x, y, w, h), n)
  gt = ... # prediction bounding box (n, (x, y, w, h))

  iou = bbox_iou(pred, gt, SDIoU=True, std_type='mean')
  ...

```

<br/>
<br/>

### References
1. Zhaohui et al, Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression, https://arxiv.org/abs/1911.08287
2. Wikipedia, Distance, https://en.wikipedia.org/wiki/Distance

<br/><div align="center">
by [seareale](https://github.com/seareale) | [KVL Lab](http://vl.knu.ac.kr) | [KNU](http://knu.ac.kr)
</div>
