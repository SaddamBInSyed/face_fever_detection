import cv2
import numpy as np

def compensate_affine_translation(x_lim, y_lim):

    x_trans = - np.min(x_lim)
    y_trans = - np.min(y_lim)

    return x_trans, y_trans

def get_rect_corners(shape, tl_corner=(0, 0)):

    """
    Get corners (x,y) coordinates of a rectangle givnes its' top left corner.

    Corners are returned as [top_left, top_right, bottom_right, bottom_left]

    """

    h = shape[0]
    w = shape[1]

    corners = tl_corner + np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h -1]])

    return corners

def warp_points(points, M):
    """
    Apply affine transformation to 2D points

        Parameters
        ----------
        points : ndarray with dtype float/double
            Array of (x, y) with shape of (N, 2).

        M : ndarray
            Affine transformation matrix of shape (2, 3).

        Returns
        -------
        points_warped : ndarray
            (N, 2) array of warped points
    """

    if type(points).__module__ != np.__name__:
        points = np.array(points)[np.newaxis, :]


    # find full affine matrix
    row_M = np.array([[0, 0, 1]])
    M = np.concatenate((M, row_M), axis=0)

    size = len(points.shape)

    for m in range(3 - size):
        points = points[np.newaxis, ...]

    points_warped = cv2.perspectiveTransform(points.astype(np.float64), M)
    points_warped = np.squeeze(points_warped)

    return points_warped


def get_outer_bounding_rect(corners):
    """
    Get bounding rectangle of (rotated) image.

        Parameters
        ----------
        corners : ndarray
            (Rotated) image (x, y) coordinates.

        Returns
        -------
        x_lim : ndarray
            x coordinates of outer bounding rectangle.
        y_lim : ndarray
            y coordinates of outer bounding rectangle.
    """

    x = corners[:, 0]
    y = corners[:, 1]

    ind_x_sorted = np.argsort(x)
    ind_y_sorted = np.argsort(y)

    x_lim = x[ind_x_sorted[[0, 3]] ]# take 2 extreme values
    y_lim = y[ind_y_sorted[[0, 3]]]

    return x_lim, y_lim


def calculate_affine_matrix(rotation_angle=0, rotation_center=(0, 0), translation=(0, 0), scale=1):
    """
    Calculate affine matrix composed of rotation, translation and scale.
    Transformation order:
        - rotation (about rotation center) and scale.
        - scaled translation

    Parameters
    ----------
    rotation_angle : float, optional
        Rotation angle in degrees.
    rotation_center : tuple, optional
        Coordinates of point about which rotation is performed.
    translation : tuple, optional
        (x, y) translation.
    scale : float, optional
        Scale factor in percents.

    Returns
    -------
    Mc : ndarray
        Compensated affine matrix of shape (2, 3)
    """

    M_rot = cv2.getRotationMatrix2D(rotation_center, rotation_angle, scale)
    translation = 1. * scale * np.array(translation)
    M_trans = np.array([[1., 0., translation[0]], [0., 1., translation[1]]])

    M = concatenate_affine_matrices(M_rot, M_trans)

    return M

def concatenate_affine_matrices(first, second):
    """
    Concatenate 2 affine matrices, each of shape (2, 3).
    """

    row = np.array([[0, 0, 1]])
    first = np.concatenate((first, row))
    second = np.concatenate((second, row))

    M = np.dot(second, first)

    M = M[:-1, :]

    return M


def cal_affine_matrix_inverse(M):

    return cv2.invertAffineTransform(M)


def warp_affine_without_crop(img, M):
    """
    Perform affine transformation such that the resulting image will not be cropped.

    Parameters
    ----------
    img : ndarray
        Image to be warped (transformed).
    M : ndarray
        Affine transformation matrix of shape (2, 3).

    Returns
    -------
    img_warped : ndarray
        Warped image
    Mc : ndarray
        Compensated affine matrix
    """

    # calculate warped image cornets
    corners = get_rect_corners(img.shape)
    corners_warped = warp_points(corners, M)

    # calculate warped image outer bounding rectangle
    x_lim_warped, y_lim_warped = get_outer_bounding_rect(corners_warped)

    # --- compensate affine transform by adding appropriate translation ---
    # calculate needed translation
    x_trans, y_trans = compensate_affine_translation(x_lim_warped, y_lim_warped)
    # calculate affine matrix that apply this translation
    M_trans = calculate_affine_matrix(translation=(x_trans, y_trans))
    # calculate compensated affine matrix
    Mc = concatenate_affine_matrices(M, M_trans)
    # ---------------------------------------------------------------------

    # calculate warped image size
    w = 1 + (x_lim_warped[1] - x_lim_warped[0]).astype(int)  # +1 since indices are zero based
    h = 1 + (y_lim_warped[1] - y_lim_warped[0]).astype(int)  # +1 since indices are zero based

    # apply transformation
    img_warped = cv2.warpAffine(img, Mc, (w, h))

    return img_warped, Mc




if __name__ == '__main__':

    img = np.arange(9, dtype=np.float32).reshape((3,3))
    # img = np.arange(12, dtype=np.float32).reshape((4,3))
    print('original')
    print(img)

    M = calculate_affine_matrix(rotation_angle=-90)

    img_warped, Mc = warp_affine_without_crop(img, M)
    print('warped')
    print(img_warped)

    x = np.array([1, 1])

    x_warped = warp_points(x, M)

    M_inv = cal_affine_matrix_inverse(M)

    y = warp_points(x_warped, M_inv)

    print('Done!')

