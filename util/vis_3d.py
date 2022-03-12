import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def display_mesh(path, verts, faces, keypoints=None, spheres_c=None, spheres_r=None, transpose=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # XYZ -> XZY
    if transpose:
        verts = verts[:, [0, 2, 1]]

    mesh = Poly3DCollection(verts[faces], alpha=0.3)
    face_color = (141 / 255, 184 / 255, 226 / 255)
    edge_color = (50 / 255, 50 / 255, 50 / 255)
    mesh.set_facecolor(face_color)
    mesh.set_edgecolor(edge_color)
    # skin_color = (238/255, 175/255, 147/255)
    # mesh.set_facecolor(skin_color)
    # mesh.set_edgecolor(skin_color)

    ax.add_collection3d(mesh)

    if keypoints is not None:
        display_keypoints('', keypoints=keypoints, ax=ax, transpose=transpose)
    if spheres_c is not None:
        display_sphere(spheres_c, spheres_r, ax=ax)
    cam_equal_aspect_3d(ax, verts, transpose=transpose)

    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def display_keypoints(path, keypoints=None, ax=None, transpose=True):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    COLOR = ['#330000', '#660000', '#b30000', '#ff0000', '#ff4d4d', '#ff9999']
    if transpose:
        keypoints = keypoints[:, [0, 2, 1]]
    for i in range(keypoints.shape[0]):
        ax.scatter(keypoints[i, 0], keypoints[i, 1], keypoints[i, 2], color='red')

    cam_equal_aspect_3d(ax, keypoints, transpose=transpose)

    if path:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)


def cam_equal_aspect_3d(ax, verts, flip_x=False, transpose=True):
    '''
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    :param ax:
    :param verts:
    :param flip_x:
    :return:
    '''
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    # min_lim, max_lim = np.min(centers - r), np.max(centers + r)
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
        # ax.set_xlim(max_lim, min_lim)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
        # ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(centers[1] - r, centers[1] + r)
    ax.set_zlim(centers[2] + r, centers[2] - r)
    # ax.set_ylim(min_lim, max_lim)
    # ax.set_zlim(max_lim, min_lim)
    if transpose:
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    # ax.view_init(5, -5)
    ax.view_init(5, -85)


def display_sphere(centers, radiuss, ax=None, transpose=True):
    num = centers.shape[0]
    if transpose:
        centers = centers[:, [0, 2, 1]]
    for index in range(num):
        center = centers[index]
        radius = radiuss[index]
        t = np.linspace(0, np.pi * 2, 20)
        s = np.linspace(0, np.pi, 20)

        t, s = np.meshgrid(t, s)
        x = np.cos(t) * np.sin(s)
        y = np.sin(t) * np.sin(s)
        z = np.cos(s)
        ax.plot_surface(x*radius + center[0], y*radius + center[1], z*radius + center[2], rstride=1, cstride=1, color='red')

# pcl N x 3 verts M x 3

def display_pcl(path, verts, pcl, faces, keypoints=None, transpose=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # XYZ -> XZY
    if transpose:
        verts = verts[:, [0, 2, 1]]
        pcl = pcl[:, [0, 2, 1]]

    verts_index = np.argmin(np.sum(np.power(pcl[:,np.newaxis,:] - verts[np.newaxis,:,:], 2),axis=-1), axis=-1)
    corr_verts = verts[verts_index, :]

    x = np.stack((pcl[:, 0], corr_verts[:, 0]), axis=-1)
    y = np.stack((pcl[:, 1], corr_verts[:, 1]), axis=-1)
    z = np.stack((pcl[:, 2], corr_verts[:, 2]), axis=-1)

    # 将数组中的前两个点进行连线
    for index in range(pcl.shape[0]):
        ax.plot(x[index], y[index], z[index], c='r')

    mesh = Poly3DCollection(verts[faces], alpha=0.3)
    face_color = (141 / 255, 184 / 255, 226 / 255)
    edge_color = (50 / 255, 50 / 255, 50 / 255)
    mesh.set_facecolor(face_color)
    mesh.set_edgecolor(edge_color)

    ax.add_collection3d(mesh)

    if keypoints is not None:
        display_keypoints('', keypoints=keypoints, ax=ax, transpose=transpose)

    cam_equal_aspect_3d(ax, verts, transpose=transpose)
    plt.show()
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()