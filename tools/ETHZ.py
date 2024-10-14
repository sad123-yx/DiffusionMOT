import os
import numpy as np
import json
from PIL import Image

DATA_PATH = '../datasets/ETHZ/'
OUT_PATH = DATA_PATH + 'annotations/'

if __name__ == '__main__':

    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)

    out_path = OUT_PATH + 'train.json'
    out = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'person'}]}
    image_cnt = 0
    ann_cnt = 0
    folder=['eth01','eth02','eth03','eth05','eth07']
    for i in range(0,len(folder)):
        img_folder=folder[i]
        img_folder_path=DATA_PATH+img_folder+'/images/'
        label_folder_path=DATA_PATH+img_folder+'/labels_with_ids/'
        img_paths=os.listdir(img_folder_path)
        label_paths=os.listdir(label_folder_path)
        temp1 = [img_folder_path + i for i in img_paths]
        temp2 = [label_folder_path + i for i in label_paths]
        print("image:",image_cnt)
        for img_path, label_path in zip(temp1, temp2):
            image_cnt += 1
            # image_path=
            im = Image.open(img_path)
            split=img_path.split('/')
            temp_file=split[3]+'/'+split[4]+'/'+split[5]
            image_info = {'file_name': temp_file,
                          'id': image_cnt,
                          'height': im.size[1],
                          'width': im.size[0]}
            out['images'].append(image_info)
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
            labels = labels0.copy()
            labels[:, 2] = image_info['width'] * (labels0[:, 2] - labels0[:, 4] / 2)
            labels[:, 3] = image_info['height'] * (labels0[:, 3] - labels0[:, 5] / 2)
            labels[:, 4] = image_info['width'] * labels0[:, 4]
            labels[:, 5] = image_info['height'] * labels0[:, 5]
            for i in range(len(labels)):
                ann_cnt += 1
                fbox = labels[i, 2:6].tolist()
                ann = {'id': ann_cnt,
                       'category_id': 1,
                       'image_id': image_cnt,
                       'track_id': -1,
                       'bbox': fbox,
                       'area': fbox[2] * fbox[3],
                       'iscrowd': 0}
                out['annotations'].append(ann)
    print('loaded train for {} images and {} samples'.format(len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'))
