import numpy as np
from menpo.shape import PointCloud
import cv2
import tensorflow as tf
from sklearn.cluster import KMeans

jaw_indices = np.arange(0, 17)
lbrow_indices = np.arange(17, 22)
rbrow_indices = np.arange(22, 27)
upper_nose_indices = np.arange(27, 31)
lower_nose_indices = np.arange(31, 36)
leye_indices = np.arange(36, 42)
reye_indices = np.arange(42, 48)
outer_mouth_indices = np.arange(48, 60)
inner_mouth_indices = np.arange(60, 68)

jaw_b1 = np.arange(0, 2)
jaw_b2 = np.arange(2, 4)
jaw_b3 = np.arange(4, 6)
jaw_b4 = np.arange(6, 8)
jaw_e1 = np.arange(8, 10)
jaw_e2 = np.arange(10, 12)
jaw_e3 = np.arange(12, 14)
jaw_e4 = np.arange(14, 16)
jaw_e5 = np.arange(16, 18)
jaw_n1 = np.arange(18, 20)
jaw_n2 = np.arange(20, 22)
jaw_n3 = np.arange(22, 24)
jaw_n4 = np.arange(24, 29)
# lbrow_jaw_cofw = np.arange(17, 22)
# rbrow_jaw_cofw = np.arange(22, 27)
# upper_jaw_cofw = np.arange(27, 31)
# lower_jaw_cofw = np.arange(31, 36)
# leye_jaw_cofw = np.arange(36, 42)
# reye_jaw_cofw = np.arange(42, 48)
# outer_mouth_jaw_cofw = np.arange(48, 60)
# inner_mouth_jaw_cofw = np.arange(60, 68)

parts_68 = (jaw_indices, lbrow_indices, rbrow_indices, upper_nose_indices,
            lower_nose_indices, leye_indices, reye_indices,
            outer_mouth_indices, inner_mouth_indices)

mirrored_parts_68 = np.hstack([
    jaw_indices[::-1], rbrow_indices[::-1], lbrow_indices[::-1],
    upper_nose_indices, lower_nose_indices[::-1],
    np.roll(reye_indices[::-1], 4), np.roll(leye_indices[::-1], 4),
    np.roll(outer_mouth_indices[::-1], 7),
    np.roll(inner_mouth_indices[::-1], 5)
])

mirrored_parts_29 = np.hstack([
    np.roll(jaw_b1[::-1], 2),np.roll(jaw_b2[::-1], 2),jaw_b4,jaw_b3,
    np.roll(jaw_e1[::-1], 2),np.roll(jaw_e2[::-1], 2),jaw_e4,jaw_e3,
    np.roll(jaw_e5[::-1], 2),np.roll(jaw_n1[::-1], 2),jaw_n2,np.roll(jaw_n3[::-1], 2),
    jaw_n4

])

def mirror_landmarks_68(lms, image_size):
    return PointCloud(abs(np.array([0, image_size[1]]) - lms.as_vector(
    ).reshape(-1, 2))[mirrored_parts_68])

def mirror_landmarks_29(lms, image_size):
    return PointCloud(abs(np.array([0, image_size[1]]) - lms.as_vector(
    ).reshape(-1, 2))[mirrored_parts_29])

def mirror_image(im):
    im = im.copy()
    im.pixels = im.pixels[..., ::-1].copy()

    for group in im.landmarks:
        lms = im.landmarks[group].lms
        if lms.points.shape[0] == 68:
            im.landmarks[group] = mirror_landmarks_68(lms, im.shape)
        if lms.points.shape[0] == 29:
            im.landmarks[group] = mirror_landmarks_29(lms, im.shape)

    return im


def mirror_image_bb(im):
    im = im.copy()
    im.pixels = im.pixels[..., ::-1]
    im.landmarks['bounding_box'] = PointCloud(abs(np.array([0, im.shape[
        1]]) - im.landmarks['bounding_box'].lms.points))
    return im


def line(image, x0, y0, x1, y1, color):
    steep = False
    if x0 < 0 or x0 >= 400 or x1 < 0 or x1 >= 400 or y0 < 0 or y0 >= 400 or y1 < 0 or y1 >= 400:
        return

    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range(int(x0), int(x1) + 1):
        t = (x - x0) / float(x1 - x0)
        y = y0 * (1 - t) + y1 * t
        if steep:
            image[x, int(y)] = color
        else:
            image[int(y), x] = color
            
def point(image, x0, y0, color):
    if x0 < 0 or x0 >= 400 or y0 < 0 or y0 >= 400:
        return
    cv2.circle(image,(y0,x0),3,(0.5,0.5,0.5),-1)
    cv2.circle(image,(y0,x0),2,(1,1,1),-1)
#    for x in range(int(x0)-2, int(x0) + 3):
#        for y in range(int(y0)-2, int(y0) + 3):
#            if((x-x0)**2+(y-y0)**2<=4):
#                image[x, y] = color
def draw_landmarks_68(img, lms):
    try:
        img = 255 * (img.copy())
        for k in range(0,17):
            cv2.circle(img, (int(lms[k, 1]), int(lms[k, 0])), 3, (0, 255, 0), -1)
        for j in range(17, 36):
            cv2.circle(img, (int(lms[j, 1]), int(lms[j, 0])), 3, (255, 0, 0), -1)
        for p in range(36, 48):
            cv2.circle(img, (int(lms[p, 1]), int(lms[p, 0])), 3, (255, 0, 255), -1)
        for q in range(48, 68):
            cv2.circle(img, (int(lms[q, 1]), int(lms[q, 0])), 3, (0, 0, 255), -1)
    except:
        pass
    return img / 255.0
def draw_landmarks_29(img, lms):
    try:
        img = 255 * (img.copy())
        for k in range(29):
            cv2.circle(img, (int(lms[k, 1]), int(lms[k, 0])), 3, (0, 255, 0), -1)
        # for j in range(17, 36):
        #     cv2.circle(img, (int(lms[j, 1]), int(lms[j, 0])), 3, (255, 0, 0), -1)
        # for p in range(36, 48):
        #     cv2.circle(img, (int(lms[p, 1]), int(lms[p, 0])), 3, (255, 0, 255), -1)
        # for q in range(48, 68):
        #     cv2.circle(img, (int(lms[q, 1]), int(lms[q, 0])), 3, (0, 0, 255), -1)
    except:
        pass
    return img / 255.0
def batch_draw_landmarks_68(imgs, lms):
    return np.array([draw_landmarks_68(img, l) for img, l in zip(imgs, lms)])
def batch_draw_29(imgs, lms):
    return np.array([draw_landmarks_29(img, l) for img, l in zip(imgs, lms)])
def draw_landmarks(img, lms):
    try:
        img = img.copy()

        for i, part in enumerate(parts_68[1:]):
            circular = []

            if i in (4, 5, 6, 7):
                circular = [part[0]]

            for p1, p2 in zip(part, list(part[1:]) + circular):
                p1, p2 = lms[p1], lms[p2]

                line(img, p2[1], p2[0], p1[1], p1[0], 1)
    except:
        pass
    return img
    
def draw_landmarks_point(img, lms):
    try:
        img = img.copy()

        for i in range(lms.shape[0]):
            point(img,lms[i][0],lms[i][1],1)
    except:
        pass
    return img


def batch_draw_landmarks(imgs, lms):
    return np.array([draw_landmarks(img, l) for img, l in zip(imgs, lms)])
def batch_draw_landmarks_point(imgs, lms):
    return np.array([draw_landmarks_point(img, l) for img, l in zip(imgs, lms)])


def get_central_crop(images, box=(6, 6)):
    _, w, h, _ = images.get_shape().as_list()

    half_box = (box[0] / 2., box[1] / 2.)

    a = slice(int((w // 2) - half_box[0]), int((w // 2) + half_box[0]))
    b = slice(int((h // 2) - half_box[1]), int((h // 2) + half_box[1]))

    return images[:, a, b, :]


def build_sampling_grid(patch_shape):
    patch_shape = np.array(patch_shape)
    patch_half_shape = np.require(np.round(patch_shape / 2), dtype=int)
    start = -patch_half_shape
    end = patch_half_shape
    sampling_grid = np.mgrid[start[0]:end[0], start[1]:end[1]]
    return sampling_grid.swapaxes(0, 2).swapaxes(0, 1)


default_sampling_grid = build_sampling_grid((30, 30))


def extract_patches(pixels, centres, sampling_grid=default_sampling_grid):
    """ Extracts patches from an image.

    Args:
        pixels: a numpy array of dimensions [width, height, channels]
        centres: a numpy array of dimensions [num_patches, 2]
        sampling_grid: (patch_width, patch_height, 2)

    Returns:
        a numpy array [num_patches, width, height, channels]
    """
    pixels = pixels.transpose(2, 0, 1)

    max_x = pixels.shape[-2] - 1
    max_y = pixels.shape[-1] - 1

    patch_grid = (sampling_grid[None, :, :, :] + centres[:, None, None, :]
                  ).astype('int32')

    X = patch_grid[:, :, :, 0].clip(0, max_x)
    Y = patch_grid[:, :, :, 1].clip(0, max_y)

    return pixels[:, X, Y].transpose(1, 2, 3, 0)

def set_patches(image, patches, patch_centers, offset=None, offset_index=None):
    r"""
    Parameters
    ----------
    patches : `ndarray` or `list`
        The values of the patches.
        A ``(n_center, n_offset, self.n_channels, patch_shape)`` `ndarray`
    patch_centers : :map:`PointCloud`
        The centers to set the patches around.
    offset : `list` or `tuple` or ``(1, 2)`` `ndarray`
        The offset to apply on the patch centers within the image.
    offset_index : `int`
        The offset index within the provided `patches` argument, thus the
        index of the second dimension from which to sample.
    Raises
    ------
    ValueError
        If pixels array is not 2D
    """
    # if self.ndim != 3:
    #     raise ValueError(
    #         "Only 2D images are supported but " "found {}".format(self.shape)
    #     )
    if offset is None:
        offset = np.zeros([1, 2], dtype=np.intp)
    # elif isinstance(offset, tuple) or isinstance(offset, list):
    #     offset = np.asarray([offset])
    # offset = np.require(offset, dtype=np.intp)
    if offset_index is None:
        offset_index = 0

    copy = image.copy()
    # set patches
    set_patch(patches, copy.pixels, patch_centers.points, offset, offset_index)
    return copy

def set_patch(patches, pixels, patch_centers, offset, offset_index):
    r"""
    Set the values of a group of patches into the correct regions of a copy
    of this image. Given an array of patches and a set of patch centers,
    the patches' values are copied in the regions of the image that are
    centred on the coordinates of the given centers.
    The patches argument can have any of the two formats that are returned
    from the `extract_patches()` and `extract_patches_around_landmarks()`
    methods. Specifically it can be:
        1. ``(n_center, n_offset, self.n_channels, patch_shape)`` `ndarray`
        2. `list` of ``n_center * n_offset`` :map:`Image` objects
    Currently only 2D images are supported.
    Parameters
    ----------
    patches : `ndarray` or `list`
        The values of the patches.
        A ``(n_center, n_offset, self.n_channels, patch_shape)`` `ndarray`
    pixels : ``(n_channels, height, width)`` `ndarray``
        Pixel array to replace the patches within
    patch_centers : :map:`PointCloud`
        The centers to set the patches around.
    offset : `list` or `tuple` or ``(1, 2)`` `ndarray`
        The offset to apply on the patch centers within the image.
    offset_index : `int`
        The offset index within the provided `patches` argument, thus the
        index of the second dimension from which to sample.
    Raises
    ------
    ValueError
        If pixels array is not 2D
    """
    if pixels.ndim != 3:
        raise ValueError(
            "Only 2D images are supported but " "found {}".format(pixels.shape)
        )

    patch_shape = patches.shape[-2:]
    # the [L]ow offset is the floor of half the patch shape
    l_r, l_c = (int(patch_shape[0] // 2), int(patch_shape[1] // 2))
    # the [H]igh offset needs to be one pixel larger if the original
    # patch was odd
    h_r, h_c = (int(l_r + patch_shape[0] % 2), int(l_c + patch_shape[1] % 2))
    for patches_with_offsets, point in zip(patches, patch_centers):
        patch = patches_with_offsets[offset_index]
        p = point + offset[0]
        p_r = int(p[0])
        p_c = int(p[1])
        pixels[:, p_r - l_r : p_r + h_r, p_c - l_c : p_c + h_c] = patch
def k_means(shapes,k,num_patches=68):
    dataMat = shapes.reshape(-1,num_patches*2)
    return KMeans(n_clusters=k,random_state=0).fit(dataMat)


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
          List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def rotate_points_tensor(points, image, angle):

    s = tf.shape(image)
    image_center = tf.to_float(s[:2]) / 2.

    # center coordinates since rotation center is supposed to be in the image center
    points_centered = points - image_center

    rot_matrix = tf.dynamic_stitch([[0], [1], [2], [3]], [tf.cos(angle), -tf.sin(angle), tf.sin(angle), tf.cos(angle)])
    rot_matrix = tf.reshape(rot_matrix, shape=[2, 2])

    points_centered_rot = tf.matmul(rot_matrix, tf.transpose(points_centered))

    return tf.transpose(points_centered_rot) + image_center

def rotate_image_tensor(image, angle):
    s = tf.shape(image)
    image_center = tf.to_float(s[:2]) / 2.

    # Coordinates of new image
    xs, ys = tf.meshgrid(tf.range(0.,tf.to_float(s[1])), tf.range(0., tf.to_float(s[0])))
    coords_new = tf.reshape(tf.stack([ys,xs], 2), [-1, 2])

    # center coordinates since rotation center is supposed to be in the image center
    coords_new_centered = tf.to_float(coords_new) - image_center

    # Perform backward transformation of the image coordinates
    rot_mat_inv = tf.stack(
        [tf.cos(angle), tf.sin(angle), -tf.sin(angle), tf.cos(angle)])
    rot_mat_inv = tf.reshape(rot_mat_inv, shape=[2, 2])
    coord_old_centered = tf.matmul(
        rot_mat_inv, tf.transpose(coords_new_centered))
    coord_old = tf.to_int32(tf.round(
        tf.transpose(coord_old_centered) + image_center))


    # Find nearest neighbor in old image
    coord_old_y, coord_old_x = tf.unstack(coord_old, axis=1)


    # Clip values to stay inside image coordinates
    outside_y = tf.logical_or(tf.greater(
        coord_old_y, s[0]-1), tf.less(coord_old_y, 0))
    outside_x = tf.logical_or(tf.greater(
        coord_old_x, s[1]-1), tf.less(coord_old_x, 0))
    outside_ind = tf.logical_or(outside_y, outside_x)


    inside_mask = tf.logical_not(outside_ind)
    inside_mask = tf.tile(tf.reshape(inside_mask, s[:2])[...,None], tf.stack([1,1,s[2]]))

    coord_old_y = tf.clip_by_value(coord_old_y, 0, s[0]-1)
    coord_old_x = tf.clip_by_value(coord_old_x, 0, s[1]-1)
    coord_flat = coord_old_y * s[1] + coord_old_x

    im_flat = tf.reshape(image, tf.stack([-1, s[2]]))
    rot_image = tf.gather(im_flat, coord_flat)
    rot_image = tf.reshape(rot_image, s)


    return tf.where(inside_mask, rot_image, tf.zeros_like(rot_image))