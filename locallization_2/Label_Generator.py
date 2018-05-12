import numpy as np
import os, csv, argparse
from PIL import Image, ImageDraw


def draw_polygon(img, coords_list):
    img = Image.fromarray(img, mode='L')
    ImageDraw.Draw(img).polygon(np.float32(coords_list), outline=128, fill=128)
    return img


def main(args):
    with open(args.csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            file_name, tl_x = row[0].split('\t')
            tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = row[1:]
            coords = [tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y]

            names = file_name.split('.')
            new_name = names[0] + '_black.' + names[1]
            print(os.path.join(args.output_path, new_name))

            i = 0
            coords_list = []
            while i < len(coords):
                coords_list.append((coords[i], coords[i+1]))
                i += 2

            img = Image.open(os.path.join(args.original_data_path, file_name), 'r').convert('L')
            img = np.array(img)
            black_img = np.zeros(img.shape)
            black_img = draw_polygon(black_img, coords_list)

            black_img.save(os.path.join(args.output_path, new_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='It is for generating label mask')
    parser.add_argument('--csv_path', default='../data/coordinates/dataset_csv_val.csv')
    parser.add_argument('--output_path', default='../data/val_labels/')
    parser.add_argument('--original_data_path', default='/home/hackathon/dataset_documents/sampleDataset/input_frame_val/')
    args = parser.parse_args()
    main(args)