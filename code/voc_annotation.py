#---------------------------------------------#
#   运行前一定要修改classes
#   如果生成的2007_train.txt里面没有目标信息
#   那么就是因为classes没有设定正确
#---------------------------------------------#
import xml.etree.ElementTree as ET
from os import getcwd

sets=['train', 'val', 'test']

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ["crazing","inclusion","pitted_surface","rolled-in_scale","patches","scratches"]
def convert_annotation(image_id, list_file):
    in_file = open('./NEU-DET/Annotations/%s.xml'%image_id, encoding='utf-8')
    tree=ET.parse(in_file)    #整个xml文件是一个tree
    root = tree.getroot()     # tree的根

    for obj in root.iter('object'):   #访问有'object'的对象  object可能不止一个
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:  #如果被标记为difficult 则不将其归为ground truth
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        # 每个object的(x1,y1,x2,y2,id)用空格进行分割,方便后面进行划分

wd = getcwd()

for image_set in sets:
    image_ids = open('NEU-DET/ImageSets/Main/%s.txt'%image_set).read().strip().split()
    list_file = open('%s.txt' % image_set, 'w')
    for image_id in image_ids:
        list_file.write('%s/NEU-DET/JPEGImages/%s.jpg'%(wd, image_id))  #存放图片地址
        convert_annotation(image_id, list_file)
        list_file.write('\n')
    list_file.close()
