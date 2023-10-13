import numpy as np

def collate_masks(mask_3d1, mask_3d2, iou_threshold=0.3):
    """Collate two masks into one.
    Also reorder mask ids to be 1 to n.
    Plus, calculate unique ROIs going forward and backward (from mask_3d1 to mask_3d2 and vice versa)
    using IoU threshold
    """
    assert len(mask_3d1.shape) == len(mask_3d2.shape) == 3
    assert mask_3d1.max() == mask_3d1.shape[2]
    assert mask_3d2.max() == mask_3d2.shape[2]
    iou_mat, ioa_mat1, ioa_mat2 = get_iou_ioa_mat(mask_3d1, mask_3d2)

    num_unique_in_mask1 = len(np.where(np.sum(iou_mat >= iou_threshold, axis=1) == 0)[0])
    num_unique_in_mask2 = len(np.where(np.sum(iou_mat >= iou_threshold, axis=0) == 0)[0])

    ind_remove_from_mask1, ind_remove_from_mask2, \
        ioa_mat1, ioa_mat2 = deal_with_multi_overlap(ioa_mat1, ioa_mat2,
                                                    mask_3d1, mask_3d2)
    ind_remove_from_mask1, ind_remove_from_mask2 \
        = deal_with_one_to_one(ioa_mat1, ioa_mat2, mask_3d1, mask_3d2,
                               ind_remove_from_mask1, ind_remove_from_mask2)
    new_mask = clean_and_merge_masks(mask_3d1, mask_3d2, 
                                     ind_remove_from_mask1,
                                     ind_remove_from_mask2)
    return new_mask, (num_unique_in_mask1, num_unique_in_mask2)


def make_3d_mask(mask_2d):
    """Make 3D mask from 2D mask.
    Also reorder mask ids to be 1 to n
    """
    ids = np.setdiff1d(np.unique(mask_2d), 0)
    mask_3d = np.zeros((mask_2d.shape[0], mask_2d.shape[1], len(ids)), dtype=np.uint16)
    for i, id_ in enumerate(ids):
        y,x = np.where(mask_2d == id_)
        mask_3d[y, x, i] = i+1
    return mask_3d


def get_iou_ioa_mat(mask_3d1, mask_3d2):
    """Get IoU and IoA matrices from two masks.
    IoU: intersection over union
    IoA: intersection over area (for each mask)
    """
    # mask_3d are ordered to have 1 to n ids
    assert len(mask_3d1.shape) == len(mask_3d2.shape) == 3
    assert mask_3d1.max() == mask_3d1.shape[2]
    assert mask_3d2.max() == mask_3d2.shape[2]

    iou_mat = np.zeros((mask_3d1.shape[2], mask_3d2.shape[2]))
    ioa_mat1 = np.zeros((mask_3d1.shape[2], mask_3d2.shape[2]))
    ioa_mat2 = np.zeros((mask_3d1.shape[2], mask_3d2.shape[2]))

    # going from mask1 to mask2
    # Collect all the roi_inds2 that are visited by roi_inds1
    visited_roi_inds2 = []
    for j in range(mask_3d1.shape[2]):
        temp_roi_ind1_map = np.sum(mask_3d1 == j + 1, axis=2)
        y, x = np.where(temp_roi_ind1_map > 0)
        
        temp_roi_ind2_visit = np.setdiff1d(np.unique(mask_3d2[y, x, :]), [0])
        visited_roi_inds2.extend(temp_roi_ind2_visit)
        for temp_roi_2 in temp_roi_ind2_visit:
            k = temp_roi_2 - 1
            temp_roi_ind2_map = np.sum(mask_3d2 == temp_roi_2, axis=2)
            intersection_map = temp_roi_ind1_map * temp_roi_ind2_map > 0
            union_map = temp_roi_ind1_map + temp_roi_ind2_map > 0
            iou_mat[j, k] = np.sum(intersection_map) / np.sum(union_map)
            ioa_mat1[j, k] = np.sum(intersection_map) / np.sum(temp_roi_ind1_map)
            ioa_mat2[j, k] = np.sum(intersection_map) / np.sum(temp_roi_ind2_map)
    # going from mask2 to mask1
    # Collect all the roi_inds1 that are not visited by roi_inds2 in the previous step
    left_roi_inds2 = np.setdiff1d(np.arange(1,mask_3d2.shape[2]), np.unique(visited_roi_inds2))
    for temp_roi_ind2 in left_roi_inds2:
        k = temp_roi_ind2 - 1
        temp_roi_ind2_map = np.sum(mask_3d2 == temp_roi_ind2, axis=2)
        y, x = np.where(temp_roi_ind2_map > 0)
        temp_roi_ind1_visit = np.setdiff1d(np.unique(mask_3d1[y, x, :]), [0])
        for temp_roi_ind1 in temp_roi_ind1_visit:
            j = temp_roi_ind1 - 1
            temp_roi_ind1_map = np.sum(mask_3d1 == temp_roi_ind1, axis=2)
            intersection_map = temp_roi_ind2_map * temp_roi_ind1_map > 0
            union_map = temp_roi_ind2_map + temp_roi_ind1_map > 0
            iou_mat[j, k] = np.sum(intersection_map) / np.sum(union_map)
            ioa_mat1[j, k] = np.sum(intersection_map) / np.sum(temp_roi_ind1_map)
            ioa_mat2[j, k] = np.sum(intersection_map) / np.sum(temp_roi_ind2_map)
    return iou_mat, ioa_mat1, ioa_mat2


def deal_with_multi_overlap(ioa_mat1, ioa_mat2, mask_3d1, mask_3d2, ioa_threshold=0.5):
    """ Deal with multi-to-multi overlap
    In case of overlap, take the ones with are closer to median (of all the ROIs)
    """
    assert len(mask_3d1.shape) == len(mask_3d2.shape) == 3
    assert mask_3d1.max() == mask_3d1.shape[2]
    assert mask_3d2.max() == mask_3d2.shape[2]
    # Only care about multi-to-one, since that was the only cases that I've seen
    # when there are multiple rois from mask2 that overlap with one roi from mask1
    multi_ind1 = np.where(np.sum(ioa_mat2 >= ioa_threshold, axis=1)>1)[0]
    # when there are multiple rois from mask1 that overlap with one roi from mask2
    multi_ind2 = np.where(np.sum(ioa_mat1 >= ioa_threshold, axis=0)>1)[0]
    # These two are assumed to be disjoint, because there was no multi-to-multi case

    area_dist1 = np.array([np.sum(mask_3d1==id) for id in range(1, mask_3d1.max()+1)])
    area_dist2 = np.array([np.sum(mask_3d2==id) for id in range(1, mask_3d2.max()+1)])
    area_dist = np.concatenate([area_dist1, area_dist2])

    ind_remove_from_mask1 = []
    ind_remove_from_mask2 = []

    for ind in multi_ind1:
        temp_overlap_roi_inds2 = np.where(ioa_mat2[ind,:] >= ioa_threshold)[0]
        # pick the one(s) with closer to median area
        percentile_to_med1 = np.abs(0.5 - np.sum(area_dist < area_dist1[ind])/len(area_dist))
        percentile_to_med2 = [np.abs(0.5 - np.sum(area_dist < area_dist2[i])/len(area_dist)) for i in temp_overlap_roi_inds2]
        if (percentile_to_med2 > percentile_to_med1).any():
            for temp_ind in temp_overlap_roi_inds2:
                ind_remove_from_mask2.append(temp_ind)
        else:
            ind_remove_from_mask1.append(ind)

    # repeat for multi_ind2
    for ind in multi_ind2:
        temp_overlap_roi_inds1 = np.where(ioa_mat1[:,ind] >= ioa_threshold)[0]
        # pick the one(s) with closer to median area
        percentile_to_med2 = np.abs(0.5 - np.sum(area_dist < area_dist2[ind])/len(area_dist))
        percentile_to_med1 = [np.abs(0.5 - np.sum(area_dist < area_dist1[i])/len(area_dist)) for i in temp_overlap_roi_inds1]
        if (percentile_to_med1 > percentile_to_med2).any():
            for temp_ind in temp_overlap_roi_inds1:
                ind_remove_from_mask1.append(temp_ind)
        else:
            ind_remove_from_mask2.append(ind)
    for ind in ind_remove_from_mask1:
        ioa_mat1[ind,:] = 0
        ioa_mat2[ind,:] = 0
    for ind in ind_remove_from_mask2:
        ioa_mat1[:,ind] = 0
        ioa_mat2[:,ind] = 0
    return ind_remove_from_mask1, ind_remove_from_mask2, ioa_mat1, ioa_mat2


def deal_with_one_to_one(ioa_mat1, ioa_mat2, mask_3d1, mask_3d2,
                        ind_remove_from_mask1, ind_remove_from_mask2,
                        ioa_threshold=0.5):
    """ Deal with one-to-one overlap
    In case of overlap, take the one with is closer to median (of all the ROIs)
    It assumes that there is no multi-to-multi overlap left
    """

    assert len(mask_3d1.shape) == len(mask_3d2.shape) == 3
    assert mask_3d1.max() == mask_3d1.shape[2]
    assert mask_3d2.max() == mask_3d2.shape[2]
    # Sweeping through one direction is enough, because only one-to-one match (if any) is left
    area_dist1 = np.array([np.sum(mask_3d1==id) for id in range(1, mask_3d1.max()+1)])
    area_dist2 = np.array([np.sum(mask_3d2==id) for id in range(1, mask_3d2.max()+1)])
    area_dist = np.concatenate([area_dist1, area_dist2])
    for ind1 in range(ioa_mat1.shape[0]):
        if (ioa_mat1[ind1,:] >= ioa_threshold).any() or (ioa_mat2[ind1,:] >= ioa_threshold).any():
            match_ind2 = np.concatenate([np.where(ioa_mat1[ind1,:] >= ioa_threshold)[0],
                                         np.where(ioa_mat2[ind1,:] >= ioa_threshold)[0]])
            match_ind2 = np.unique(match_ind2)
            percentile_to_med1 = np.abs(0.5 - np.sum(area_dist < area_dist1[ind1])/len(area_dist))
            percentile_to_med2 = np.abs(0.5 - np.sum(area_dist < area_dist2[match_ind2])/len(area_dist))
            if percentile_to_med1 < percentile_to_med2:
                ind_remove_from_mask2.append(match_ind2)
            else:
                ind_remove_from_mask1.append(ind1)
    return ind_remove_from_mask1, ind_remove_from_mask2


def clean_and_merge_masks(mask_3d1, mask_3d2, ind_remove_from_mask1, ind_remove_from_mask2):
    """Clean and merge two masks.
    Also reorder mask ids to be 1 to n
    """
    assert len(mask_3d1.shape) == len(mask_3d2.shape) == 3
    assert mask_3d1.max() == mask_3d1.shape[2]
    assert mask_3d2.max() == mask_3d2.shape[2]
    # just to be sure (ids1 = range(1, mask_3d1.max()+1)
    ids1 = np.setdiff1d(np.unique(mask_3d1), 0)
    ids2 = np.setdiff1d(np.unique(mask_3d2), 0)
    for ind1 in ind_remove_from_mask1:
        mask_3d1[mask_3d1==ids1[ind1]] = 0
    for ind2 in ind_remove_from_mask2:
        mask_3d2[mask_3d2==ids2[ind2]] = 0
    mask_3d1, _ = reorder_mask(mask_3d1)
    mask_3d2, _ = reorder_mask(mask_3d2, startswith=mask_3d1.max()+1)
    mask_3d_merged = np.dstack([mask_3d1, mask_3d2])
    return mask_3d_merged


def reorder_mask(mask_3d, startswith=1):
    """Reorder mask ids to be 1 to n
    """
    ids = np.setdiff1d(np.unique(mask_3d), 0)
    new_mask = np.zeros((*mask_3d.shape[:2], len(ids)), dtype=np.uint16)
    for i, id in enumerate(ids):
        y, x = np.where(mask_3d == id)[:2]
        new_mask[y, x, i] = i + startswith
    new_ids = np.setdiff1d(np.unique(new_mask), 0)
    return new_mask, new_ids


