import os
import json
import argparse
from shutil import copyfile


def create_yolo_dataset(anno_file, label_map_file, images_dir, output_dir):
    with open(anno_file, 'r') as f:
        annotations = json.load(f)

    with open(label_map_file, 'r') as f:
        label_map = json.load(f)

    images = annotations['images']
    annotations = annotations['annotations']

    # Create YOLO directories
    train_images_dir = os.path.join(output_dir, 'dataset', 'train', 'images')
    train_labels_dir = os.path.join(output_dir, 'dataset', 'train', 'labels')
    test_images_dir = os.path.join(output_dir, 'dataset', 'test', 'images')
    test_labels_dir = os.path.join(output_dir, 'dataset', 'test', 'labels')

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)

    for image_info in images:
        image_id = image_info['id']
        file_name = image_info['file_name']
        image_width = image_info['width']
        image_height = image_info['height']

        label_file_content = ""

        for annotation in annotations:
            if annotation['image_id'] == image_id:
                category_id = annotation['category_id']
                category_name = list(label_map.keys())[list(label_map.values()).index(category_id)]

                bbox = annotation['bbox']
                x_center = (bbox[0] + bbox[2] / 2) / image_width
                y_center = (bbox[1] + bbox[3] / 2) / image_height
                width = bbox[2] / image_width
                height = bbox[3] / image_height

                label_file_content += f"{label_map[category_name]} {x_center} {y_center} {width} {height}\n"

        if label_file_content:
            # Write YOLO label file
            label_file_path = os.path.join(test_labels_dir, file_name.replace('.jpg', '.txt')).replace('rtsd-frames/', '')
            os.makedirs(os.path.dirname(label_file_path), exist_ok=True)
            with open(label_file_path, 'w') as label_file:
                label_file.write(label_file_content)

            # Copy image to YOLO images directory
            image_path = os.path.join(images_dir, file_name.replace('rtsd-frames/', ''))
            copyfile(image_path, os.path.join(test_images_dir, file_name.replace('rtsd-frames/', '')))

    # Create data.yaml
    data_yaml_content = f"train: {os.path.join('train', 'images')}\nval: {os.path.join('test', 'images')}\n"
    data_yaml_content += f"nc: {len(label_map)}\nnames: {list(label_map.keys())}"
    with open(os.path.join(output_dir, 'dataset', 'data.yaml'), 'w') as data_yaml:
        data_yaml.write(data_yaml_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert annotations to YOLO format.')
    parser.add_argument('--anno-file', dest='anno_file', default='archive/val_anno.json', help='Path to annotation file')
    parser.add_argument('--label-map-file', dest='label_map_file', default='archive/label_map.json', help='Path to label map file')
    parser.add_argument('--images-dir', dest='images_dir', default='archive/rtsd-frames/rtsd-frames', help='Path to images directory')
    parser.add_argument('--output-dir', dest='output_dir', default='output', help='Output directory for YOLO dataset')

    args = parser.parse_args()

    create_yolo_dataset(args.anno_file, args.label_map_file, args.images_dir, args.output_dir)
