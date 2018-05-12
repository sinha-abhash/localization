import cv2
import os, csv
import numpy as np


def main():
    res = 224
    img_path = '/home/hackathon/dataset_documents/sampleDataset/input_frame_train/'
    coords_csv = ['../data/coordinates/magazine_csv.csv', '../data/coordinates/dataset_csv.csv', '../data/coordinates/letter_csv.csv']
    img_output_path = '../data/train_image'
    files = os.listdir(img_path)
    count = 0
    with open('../data/coordinates/translated_coords.csv', 'w') as csv_write:
        csv_writer = csv.writer(csv_write)
        for f in files:
            img = cv2.imread(os.path.join(img_path, f))
            #img = np.array(img)
            found = False
            resize_img = cv2.resize(img, (res, res))
            cv2.imwrite(os.path.join(img_output_path, f),resize_img)
            for csv_f in coords_csv:
                with open(csv_f, 'r') as csv_file:
                    reader = csv.reader(csv_file)
                    for rows in reader:
                        filename, tl_x = rows[0].split('\t')
                        if f == filename:
                            count += 1
                            found = True
                            tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = rows[1:]
                            coords = [tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y]
                            coords_orig = coords
                            coords = np.reshape(coords, (4,2))
                            coords = np.float32(coords)
                            for i in range(4):
                                coords[i][0] = (res / img.shape[1]) * coords[i][0]
                                coords[i][1] = (res / img.shape[0]) * coords[i][1]
                                #cv2.circle(resize_img, (coords[i][0], coords[i][1]), radius=3, color=(0,255,255), thickness=1, lineType=8)
                            coords = np.reshape(coords, (1,8))
                            entry = [f]
                            for i in coords:
                                entry.extend(i)
                            csv_writer.writerow(entry)
                            break
                if found:
                    break

    print("Number of entries found: %d and Number of files in dir: %d" %(count, len(files)))

if __name__ == '__main__':
    main()