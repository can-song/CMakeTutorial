"""
因为ILSVRC2012_img_val文件中的图片没有按标签放到制定的文件夹中，故该代码根据ILSVRC2012_devkit_t12中的标签信息
将ILSVRC2012_img_val文件中的图片分类放到制定的文件夹中，方便使用dataloader进行加载。
"""
import os
import scipy.io
import shutil
def move_valimg(val_dir='./data/imagenet/val', devkit_dir='./data/ILSVRC2012_devkit_t12'):
    """
    move valimg to correspongding folders.
    val_id(start from 1) -> ILSVRC_ID(start from 1) -> WIND
    organize like:
    /val
       /n01440764
           images
       /n01443537
           images
        .....
    """
    # load synset, val ground truth and val images list
    synset = scipy.io.loadmat(os.path.join(devkit_dir, 'data', 'meta.mat'))
    ground_truth = open(os.path.join(devkit_dir, 'data', 'ILSVRC2012_validation_ground_truth.txt'))
    lines = ground_truth.readlines()
    labels = [int(line[:-1]) for line in lines]
    root, _, filenames = next(os.walk(val_dir))
    for filename in filenames:
        # val image name -> ILSVRC ID -> WIND
        val_id = int(filename.split('.')[0].split('_')[-1])
        ILSVRC_ID = labels[val_id - 1]
        WIND = synset['synsets'][ILSVRC_ID - 1][0][1][0]
        print("val_id:%d, ILSVRC_ID:%d, WIND:%s" % (val_id, ILSVRC_ID, WIND))
        # move val images
        output_dir = os.path.join(root, WIND)
        if os.path.isdir(output_dir):
             pass
        else:
            os.mkdir(output_dir)
        shutil.move(os.path.join(root, filename), os.path.join(output_dir, filename))

if __name__ == '__main__':
    val_dir = './ImageNet2012/ILSVRC2012_img_val'
    devkit_dir = './ImageNet/data/ImageNet2012/ILSVRC2012_devkit_t12'
    move_valimg(val_dir, devkit_dir)