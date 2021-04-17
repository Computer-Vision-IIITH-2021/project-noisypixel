from pykdtree.kdtree import KDTree
import numpy as np

pc_path1 = '/home/madhvi/Documents/CV Project/data/subset/ShapeNet/02691156/1ac29674746a0fc6b87697d3904b168b/pointcloud.npz'
pc_path2 = '/home/madhvi/Documents/CV Project/data/subset/ShapeNet/02691156/1ac29674746a0fc6b87697d3904b168b/pointcloud.npz'
point_path1 = '/home/madhvi/Documents/CV Project/data/subset/ShapeNet/02691156/1ac29674746a0fc6b87697d3904b168b/points.npz'
point_path2 = '/home/madhvi/Documents/CV Project/data/subset/ShapeNet/02691156/1ac29674746a0fc6b87697d3904b168b/points.npz'

pc_data1 = np.load(pc_path1)
pc_data2 = np.load(pc_path2)
points_data1 = np.load(point_path1)
points_data2 = np.load(point_path2)

pointcloud = pc_data1['points']
pointcloud_gt = pc_data2['points']
normals = pc_data1['normals']
normals_gt = pc_data2['normals']
occ_1 = points_data1['occupancies']
occ_2 = points_data2['occupancies']


def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.
    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou


def compute_separation(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    sepr, ind = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[ind] * normals_src).sum(axis=-1)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return sepr, normals_dot_product


def eval_pointcloud(pointcloud, pointcloud_gt,
                        normals, normals_gt, occ1, occ2):
        ''' 
        Evaluates a point cloud.
        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_gt (numpy array): ground truth point cloud
            normals (numpy array): predicted normals
            normals_gt (numpy array): ground truth normals
        '''
        # Return maximum losses if pointcloud is empty

        empty_point_dict = {
		    'completeness': np.sqrt(3),
		    'accuracy': np.sqrt(3),
		    'completeness2': 3,
		    'accuracy2': 3,
		    'chamfer': 6,
		}

		empty_normal_dict = {
		    'normals completeness': -1.,
		    'normals accuracy': -1.,
		    'normals': -1.,
		}

        if pointcloud.shape[0] == 0:
            print('Empty pointcloud / mesh detected!')
            out_dict = empty_point_dict
            if normals is not None and normals_tgt is not None:
                out_dict.update(empty_normal_dict)
            return out_dict

        pointcloud = np.asarray(pointcloud)
        pointcloud_gt = np.asarray(pointcloud_gt)

        # Completeness: how far are the points of the groundtruth point cloud
        # from the predicted point cloud
        completeness, normal_completeness = compute_separation(
            pointcloud_gt, normals_gt, pointcloud, normals
        )
        completeness_sq = completeness**2

        completeness = completeness.mean()
        completeness_sq = completeness_sq.mean()
        normal_completeness = normal_completeness.mean()

        # Accuracy: how far are the points of the predicted pointcloud
        # from the groundtruth pointcloud
        accuracy, normal_accuracy = compute_separation(
            pointcloud, normals, pointcloud_gt, normals_gt
        )
        accuracy_sq = accuracy**2

        accuracy = accuracy.mean()
        accuracy_sq = accuracy_sq.mean()
        normal_accuracy = normal_accuracy.mean()

        # Chamfer distance
        chamferL2 = 0.5 * (completeness_sq + accuracy_sq)
        normals_correction = (
            0.5 * normal_completeness + 0.5 * normal_accuracy
        )
        chamferL1 = 0.5 * (completeness + accuracy)
        
        occupancy_iou = compute_iou(occ1, occ2)

        out_dict = {
            'completeness': completeness,
            'accuracy': accuracy,
            'normals completeness': normal_completeness,
            'normals accuracy': normal_accuracy,
            'normals': normals_correction,
            'completeness_sq': completeness_sq,
            'accuracy_sq': accuracy_sq,
            'chamfer-L2': chamferL2,compute_iou(occ1, occ2)
            'chamfer-L1': chamferL1,
            'iou': occupancy_iou
        }

        return out_dict