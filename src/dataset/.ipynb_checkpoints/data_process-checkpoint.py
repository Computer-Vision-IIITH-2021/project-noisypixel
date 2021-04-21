import numpy as np
import pandas as pd
import h5py
import os
import skimage.io as sio
import tqdm
import argparse
import pickle as pkl

def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))

def load_dict_from_hdf5(filename):
    """
    ....
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

def load_data(path):   
    # Load the pointcloud.npz and points.npz file
    pc_file = np.load(os.path.join(path, "pointcloud.npz"))
    points_file = np.load(os.path.join(path, "points.npz"))
    
    # create image placeholder and camera data placeholder
    img_data = []
    cam_data = None
    
    # Load images
    for imx in os.listdir(os.path.join(path, "img_choy2016")):
        current = os.path.join(path, "img_choy2016", imx)
        if 'npz' in imx:
            cam_data = np.load(current)
        else:
            img_current = sio.imread(current)
            if img_current.ndim == 2:
                img_current = np.stack([img_current, img_current, img_current], axis=-1)
            img_data.append(img_current)
    img_data = np.asarray(img_data)
    
    all_data = {
        'images': img_data,
        'camera': dict(cam_data),
        'points': dict(points_file),
        'pointcloud': dict(pc_file)
    }
    
    return all_data

def main(args):
    data_root = args.dataroot
    dataset_dir = args.output
    
    # Create the output folder
    os.makedirs(os.path.join(dataset_dir, "hdf_data"), exist_ok=True)
    save_path = os.path.join(dataset_dir, "hdf_data")
    
    file_lists = {
        'train.lst': [],
        'test.lst': [],
        'val.lst': []
    }
    
    # iterate over each class in the dataset
    for cid in os.listdir(data_root):
        # Get the path to each object and list of objects
        objs_path = os.path.join(data_root, cid)
        if "metadata" in cid.lower():
            continue
        obj_list = os.listdir(objs_path)
        
        # iterate over each object in the dataset class
        for obx in tqdm.tqdm(obj_list):
            current_path = os.path.join(objs_path, obx)
            new_filename = "{}_{}.h5".format(cid, obx)

            try:
                # If possible, load the object and it's propertiess
                data_current = load_data(current_path)
                
                # Save the output into the HDF5 file at the output location
                save_dict_to_hdf5(data_current, os.path.join(save_path, new_filename))
            except:
                # Print the file name for error logs
                if obx.lower() in ["train.lst", "test.lst", "val.lst"]:
                    # read each file
                    f = open(current_path, 'r')
                    flist = ["{}_{}.h5".format(cid, yx) for yx in f.read().split()]
                    f.close()
                    
                    # Append to file lists
                    file_lists[obx] += flist
                else:
                    print("Error at {}-{}".format(cid, obx))
    
    # Now save the file lists as well
    for kx in file_lists.keys():
        # Get each file list and save
        print("Processing list for {}".format(kx))
        flist = "\n".join(file_lists[kx])
        f = open(os.path.join(save_path, kx), 'w')
        f.write(flist)
        f.close()
    print("Saved data with train-test-val splits...")


if __name__ == "__main__":
    # Create the argument parser and parse the script parameters
    parser = argparse.ArgumentParser(description='Process dataset to create HDF5 data file for each object')
    parser.add_argument('--dataroot', action='store', type=str, help="dataset path for the preprocessed shapenet files")
    parser.add_argument('--output', action='store', type=str, help="output data folder to save the dataset")
    
    args = parser.parse_args()
    
    # Run the main function
    main(args)