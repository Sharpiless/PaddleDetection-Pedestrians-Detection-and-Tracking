# PaddleDetection训练单/多镜头行人追踪模型

## 更新直接提交比赛版本：

https://aistudio.baidu.com/aistudio/projectdetail/1411754

## 项目效果：

项目AI Studio：https://aistudio.baidu.com/aistudio/projectdetail/1563160

视频地址：https://www.bilibili.com/video/BV1mz4y1y79G/


```python
!zip code.zip -r work/PaddleDetection-release-2.0-rc/ 
```

## 简介

PaddleDetection飞桨目标检测开发套件，旨在帮助开发者更快更好地完成检测模型的组建、训练、优化及部署等全开发流程。

PaddleDetection模块化地实现了多种主流目标检测算法，提供了丰富的数据增强策略、网络模块组件（如骨干网络）、损失函数等，并集成了模型压缩和跨平台高性能部署能力。

经过长时间产业实践打磨，PaddleDetection已拥有顺畅、卓越的使用体验，被工业质检、遥感图像检测、无人巡检、新零售、互联网、科研等十多个行业的开发者广泛应用。

![](https://ai-studio-static-online.cdn.bcebos.com/580ea147b8ce490189657693d9473e6d750468f787d14c5fbdcc5f515de19bec)


![](https://ai-studio-static-online.cdn.bcebos.com/c6821f0c46fe4d81a072cb39c2a8c12b998fe25679d64d3fbf74dfa83bad1b99)


## 解压数据集：

该项目数据集使用COCO数据集中的行人部分。


```python
!unzip -oq data/data7122/train2017.zip -d ./
!unzip -oq data/data7122/val2017.zip -d ./
!unzip -oq data/data7122/annotations_trainval2017.zip -d ./
```


```python
!pip install pycocotools
!pip install scikit-image
```

    Successfully installed PyWavelets-1.1.1 numpy-1.20.1 scikit-image-0.18.1 tifffile-2021.2.1



```python
!rm -rf VOCData/
```


```python
!mkdir VOCData/
!mkdir VOCData/images/
!mkdir VOCData/annotations/
```


```python
!pip install pycocotools
!pip install scikit-image
```

    Successfully installed PyWavelets-1.1.1 numpy-1.20.1 scikit-image-0.18.1 tifffile-2021.3.4


## 转换数据集格式：

将COCO中的行人类别提取出来并转换为VOC格式。


```python
from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
 
# 保存路径
savepath = "VOCData/"
img_dir = savepath + 'images/'
anno_dir = savepath + 'annotations/'
datasets_list=['train2017', 'val2017']
 
classes_names = ['person']
# 读取COCO数据集地址  Store annotations and train2014/val2014/... in this folder
dataDir = './'
 
headstr = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""
 
tailstr = '''\
</annotation>
'''
 
 
# if the dir is not exists,make it,else delete it
def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)
 
 
mkr(img_dir)
mkr(anno_dir)
 
 
def id2name(coco):
    classes = dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']] = cls['name']
    return classes
 
 
def write_xml(anno_path, head, objs, tail):
    f = open(anno_path, "w")
    f.write(head)
    for obj in objs:
        f.write(objstr % (obj[0], obj[1], obj[2], obj[3], obj[4]))
    f.write(tail)
 
 
def save_annotations_and_imgs(coco, dataset, filename, objs):
    # eg:COCO_train2014_000000196610.jpg-->COCO_train2014_000000196610.xml
    anno_path = anno_dir + filename[:-3] + 'xml'
    img_path = dataDir + dataset + '/' + filename
    dst_imgpath = img_dir + filename
 
    img = cv2.imread(img_path)
    if (img.shape[2] == 1):
        print(filename + " not a RGB image")
        return
    shutil.copy(img_path, dst_imgpath)
 
    head = headstr % (filename, img.shape[1], img.shape[0], img.shape[2])
    tail = tailstr
    write_xml(anno_path, head, objs, tail)
 
 
def showimg(coco, dataset, img, classes, cls_id, show=True):
    global dataDir
    I = Image.open('%s/%s/%s' % (dataDir, dataset, img['file_name']))
    # 通过id，得到注释的信息
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    anns = coco.loadAnns(annIds)
    # coco.showAnns(anns)
    objs = []
    for ann in anns:
        class_name = classes[ann['category_id']]
        if class_name in classes_names:
            if 'bbox' in ann:
                bbox = ann['bbox']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                obj = [class_name, xmin, ymin, xmax, ymax]
                objs.append(obj)
 
    return objs
 
 
for dataset in datasets_list:
    # ./COCO/annotations/instances_train2014.json
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataset)
 
    # COCO API for initializing annotated data
    coco = COCO(annFile)
    '''
    COCO 对象创建完毕后会输出如下信息:
    loading annotations into memory...
    Done (t=0.81s)
    creating index...
    index created!
    至此, json 脚本解析完毕, 并且将图片和对应的标注数据关联起来.
    '''
    # show all classes in coco
    classes = id2name(coco)
    print(classes)
    # [1, 2, 3, 4, 6, 8]
    classes_ids = coco.getCatIds(catNms=classes_names)
    print(classes_ids)
    for cls in classes_names:
        # Get ID number of this class
        cls_id = coco.getCatIds(catNms=[cls])
        img_ids = coco.getImgIds(catIds=cls_id)
        # imgIds=img_ids[0:10]
        for imgId in tqdm(img_ids):
            img = coco.loadImgs(imgId)[0]
            filename = img['file_name']
            objs = showimg(coco, dataset, img, classes, classes_ids, show=False)
            save_annotations_and_imgs(coco, dataset, filename, objs)
```


```python
import os
from tqdm import tqdm
from shutil import move

out_base = 'images'
img_base = 'VOCData/images/'
xml_base = 'VOCData/annotations/'

if not os.path.exists(out_base):
    os.mkdir(out_base)

for img in tqdm(os.listdir(img_base)):
    xml = img.replace('.jpg', '.xml')
    src_img = os.path.join(img_base, img)
    src_xml = os.path.join(xml_base, xml)
    dst_img = os.path.join(out_base, img)
    dst_xml = os.path.join(out_base, xml)
    if os.path.exists(src_img) and os.path.exists(src_xml):
        move(src_img, dst_img)
        move(src_xml, dst_xml)
```

    
      0%|          | 0/4553 [00:00<?, ?it/s][A
    100%|██████████| 4553/4553 [00:00<00:00, 23619.52it/s][A



```python
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET


def extract_xml(infile):
    tree = ET.parse(infile)
    root = tree.getroot()
    size = root.find('size')
    classes = []
    for obj in root.iter('object'):
        cls_ = obj.find('name').text
        classes.append(cls_)
    return classes


if __name__ == '__main__':

    base = 'VOCData/annotations/'
    Xmls = sorted([v for v in os.listdir(base) if v.endswith('.xml')])

    print('-[INFO] total:', len(Xmls))

    labels = {'person': 0}

    for xml in Xmls:
        infile = os.path.join(base, xml)
        cls_ = extract_xml(infile)
        for c in cls_:
            if not c in labels:
                print(infile, c)
                raise
            labels[c] += 1

    for k, v in labels.items():
        print('-[Count] {} total:{} per:{}'.format(k, v, v/len(Xmls)))
```

    -[INFO] total: 33328
    -[Count] person total:136140 per:4.084853576572251



```python
!rm -rf val2017/
!rm -rf train2017/
```


```python
%cd work/
```

    /home/aistudio/work



```python
!unzip PaddleDetection-release-2.0-rc.zip
```


```python
%cd PaddleDetection-release-2.0-rc/
```

    /home/aistudio/work/PaddleDetection-release-2.0-rc



```python
!mkdir VOCData
!mv ../VOCData/* VOCData/
```

## 生成数据索引文件：


```python
import os
from tqdm import tqdm
from random import shuffle

dataset = 'VOCData/'
train_txt = os.path.join(dataset, 'trainval.txt')
val_txt = os.path.join(dataset, 'val.txt')
lbl_txt = os.path.join(dataset, 'label_list.txt')

classes = [
        "person"
    ]

with open(lbl_txt, 'w') as f:
    for l in classes:
        f.write(l+'\n')

xml_base = 'Annotations'
img_base = 'images'

xmls = [v for v in os.listdir(os.path.join(dataset, xml_base)) if v.endswith('.xml')]
shuffle(xmls)

split = int(0.9 * len(xmls))

with open(train_txt, 'w') as f:
    for x in tqdm(xmls[:split]):
        m = x[:-4]+'.jpg'
        xml_path = os.path.join(xml_base, x)
        img_path = os.path.join(img_base, m)
        f.write('{} {}\n'.format(img_path, xml_path))
    
with open(val_txt, 'w') as f:
    for x in tqdm(xmls[split:]):
        m = x[:-4]+'.jpg'
        xml_path = os.path.join(xml_base, x)
        img_path = os.path.join(img_base, m)
        f.write('{} {}\n'.format(img_path, xml_path))
```

    
    100%|██████████| 29995/29995 [00:00<00:00, 309670.92it/s]
    
    100%|██████████| 3333/3333 [00:00<00:00, 294208.59it/s]



```python
!mv VOCData/* dataset/voc/
```

## 训练模型：

该部分使用ppyolo算法，算法简介可以看我的博客：

https://blog.csdn.net/weixin_44936889/article/details/107560168


```python
!python tools/train.py -c ppyolo_voc.yml --eval -o use_gpu=true 
```

## 检测效果：

![](https://ai-studio-static-online.cdn.bcebos.com/d918d9c3be954c9eaa99b03dc0be10e63cce72d3b800484e96a32016787bab98)


## 目标追踪算法：

该部分算法使用Sort算法。




```python
from PIL import Image
import dlib
import cv2
import time


def update_tracker(target_detector, image, task, roi=None):

    if roi is None:
        roi = (0, 0, image.shape[1], image.shape[0])

    if target_detector.frameCounter > 2e+4:
        target_detector.frameCounter = 0

    carIDtoDelete = []

    flag = not target_detector.carDirections is None

    for carID in target_detector.carTracker.keys():
        trackingQuality = target_detector.carTracker[carID].update(image)

        if trackingQuality < 8:
            carIDtoDelete.append(carID)

    for carID in carIDtoDelete:
        target_detector.carTracker.pop(carID, None)
        target_detector.carLocation1.pop(carID, None)
        target_detector.carLocation2.pop(carID, None)
        if carID in target_detector.carIllegals:
            target_detector.carIllegals.remove(carID)
        if flag:
            target_detector.carDirections.pop(carID, None)

    if not (target_detector.frameCounter % target_detector.stride):

        if task == 0:
            cars = target_detector.carCascade.detect_for_cars(image)
        elif task == 1:
            cars = target_detector.carCascade.detect_for_people(image)
        elif task == 2:
            cars = target_detector.carCascade.detect_for_people_cars(
                image, roi)
        else:
            cars = target_detector.carCascade.detect_for_all(image)

        for (x1, y1, x2, y2, cls_id) in cars:

            if flag:
                ROI = image[y1:y2, x1:x2]
                ROI = Image.fromarray(ROI[:, :, ::-1])
                car_direction = target_detector.classifier.predict(ROI)

            x = int(x1)
            y = int(y1)
            w = int(x2-x1)
            h = int(y2-y1)

            x_bar = x + 0.5 * w
            y_bar = y + 0.5 * h

            matchCarID = None

            for carID in target_detector.carTracker.keys():
                trackedPosition = target_detector.carTracker[carID].get_position(
                )

                t_x = int(trackedPosition.left())
                t_y = int(trackedPosition.top())
                t_w = int(trackedPosition.width())
                t_h = int(trackedPosition.height())

                t_x_bar = t_x + 0.5 * t_w
                t_y_bar = t_y + 0.5 * t_h

                if t_x <= x_bar <= (t_x + t_w) and t_y <= y_bar <= (t_y + t_h):
                    if x <= t_x_bar <= (x + w) and y <= t_y_bar <= (y + h):
                        matchCarID = carID

            if matchCarID is None:

                tracker = dlib.correlation_tracker()
                tracker.start_track(
                    image, dlib.rectangle(x, y, x + w, y + h))

                target_detector.carTracker[target_detector.currentCarID] = tracker
                target_detector.carLocation1[target_detector.currentCarID] = [
                    x, y, w, h]
                if flag:
                    target_detector.carDirections[target_detector.currentCarID] = car_direction

                target_detector.currentCarID = target_detector.currentCarID + 1

    for carID in target_detector.carTracker.keys():
        trackedPosition = target_detector.carTracker[carID].get_position()

        t_x = int(trackedPosition.left())
        t_y = int(trackedPosition.top())
        t_w = int(trackedPosition.width())
        t_h = int(trackedPosition.height())

        if carID in target_detector.carIllegals:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        # cv2.rectangle(image, (t_x, t_y), (t_x + t_w,
        #                                   t_y + t_h), color, 4)
        point = [t_x, t_y, t_x + t_w, t_y + t_h]
        plot_one_box(point, image, color, label='ID:'+str(carID))
        target_detector.carLocation2[carID] = [t_x, t_y, t_w, t_h]

    if roi:
        cv2.rectangle(image, (roi[0], roi[1]),
                      (roi[2], roi[3]), (0, 0, 255), 8)

    return image

```

## 最终效果：

视频地址：https://www.bilibili.com/video/BV1mz4y1y79G/


