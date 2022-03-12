# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-fixme[21]: Could not find name `_C` in `pytorch3d`.
from pytorch3d import _C
from pytorch3d.structures import Meshes, Pointclouds
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch

"""
This file defines distances between meshes and pointclouds.
The functions make use of the definition of a distance between a point and
an edge segment or the distance of a point and a triangle (face).

The exact mathematical formulations and implementations of these
distances can be found in `csrc/utils/geometry_utils.cuh`.
"""


# PointFaceDistance
class _PointFaceDistance(Function):
    """
    Torch autograd Function wrapper PointFaceDistance Cuda implementation
    """

    @staticmethod
    def forward(ctx, points, points_first_idx, tris, tris_first_idx, max_points):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first point
                index in each example in the batch
            tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
                triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
            tris_first_idx: LongTensor of shape `(N,)` indicating the first face
                index in each example in the batch
            max_points: Scalar equal to maximum number of points in the batch
        Returns:
            dists: FloatTensor of shape `(P,)`, where `dists[p]` is the squared
                euclidean distance of `p`-th point to the closest triangular face
                in the corresponding example in the batch
            idxs: LongTensor of shape `(P,)` indicating the closest triangular face
                in the corresponding example in the batch.

            `dists[p]` is
            `d(points[p], tris[idxs[p], 0], tris[idxs[p], 1], tris[idxs[p], 2])`
            where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
            face `(v0, v1, v2)`

        """
        dists, idxs = _C.point_face_dist_forward(
            points, points_first_idx, tris, tris_first_idx, max_points
        )
        ctx.save_for_backward(points, tris, idxs)
        return dists

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        grad_points, grad_tris = _C.point_face_dist_backward(
            points, tris, idxs, grad_dists
        )
        return grad_points, None, grad_tris, None, None


# pyre-fixme[16]: `_PointFaceDistance` has no attribute `apply`.
point_face_distance = _PointFaceDistance.apply


# FacePointDistance
class _FacePointDistance(Function):
    """
    Torch autograd Function wrapper FacePointDistance Cuda implementation
    """

    @staticmethod
    def forward(ctx, points, points_first_idx, tris, tris_first_idx, max_tris):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first point
                index in each example in the batch
            tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
                triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
            tris_first_idx: LongTensor of shape `(N,)` indicating the first face
                index in each example in the batch
            max_tris: Scalar equal to maximum number of faces in the batch
        Returns:
            dists: FloatTensor of shape `(T,)`, where `dists[t]` is the squared
                euclidean distance of `t`-th trianguar face to the closest point in the
                corresponding example in the batch
            idxs: LongTensor of shape `(T,)` indicating the closest point in the
                corresponding example in the batch.

            `dists[t] = d(points[idxs[t]], tris[t, 0], tris[t, 1], tris[t, 2])`,
            where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
            face `(v0, v1, v2)`.
        """
        dists, idxs = _C.face_point_dist_forward(
            points, points_first_idx, tris, tris_first_idx, max_tris
        )
        ctx.save_for_backward(points, tris, idxs)
        return dists

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        grad_points, grad_tris = _C.face_point_dist_backward(
            points, tris, idxs, grad_dists
        )
        return grad_points, None, grad_tris, None, None


# pyre-fixme[16]: `_FacePointDistance` has no attribute `apply`.
face_point_distance = _FacePointDistance.apply


# PointEdgeDistance
class _PointEdgeDistance(Function):
    """
    Torch autograd Function wrapper PointEdgeDistance Cuda implementation
    """

    @staticmethod
    def forward(ctx, points, points_first_idx, segms, segms_first_idx, max_points):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first point
                index for each example in the mesh
            segms: FloatTensor of shape `(S, 2, 3)` of edge segments. The `s`-th
                edge segment is spanned by `(segms[s, 0], segms[s, 1])`
            segms_first_idx: LongTensor of shape `(N,)` indicating the first edge
                index for each example in the mesh
            max_points: Scalar equal to maximum number of points in the batch
        Returns:
            dists: FloatTensor of shape `(P,)`, where `dists[p]` is the squared
                euclidean distance of `p`-th point to the closest edge in the
                corresponding example in the batch
            idxs: LongTensor of shape `(P,)` indicating the closest edge in the
                corresponding example in the batch.

            `dists[p] = d(points[p], segms[idxs[p], 0], segms[idxs[p], 1])`,
            where `d(u, v0, v1)` is the distance of point `u` from the edge segment
            spanned by `(v0, v1)`.
        """
        dists, idxs = _C.point_edge_dist_forward(
            points, points_first_idx, segms, segms_first_idx, max_points
        )
        ctx.save_for_backward(points, segms, idxs)
        return dists

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, segms, idxs = ctx.saved_tensors
        grad_points, grad_segms = _C.point_edge_dist_backward(
            points, segms, idxs, grad_dists
        )
        return grad_points, None, grad_segms, None, None


# pyre-fixme[16]: `_PointEdgeDistance` has no attribute `apply`.
point_edge_distance = _PointEdgeDistance.apply


# EdgePointDistance
class _EdgePointDistance(Function):
    """
    Torch autograd Function wrapper EdgePointDistance Cuda implementation
    """

    @staticmethod
    def forward(ctx, points, points_first_idx, segms, segms_first_idx, max_segms):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first point
                index for each example in the mesh
            segms: FloatTensor of shape `(S, 2, 3)` of edge segments. The `s`-th
                edge segment is spanned by `(segms[s, 0], segms[s, 1])`
            segms_first_idx: LongTensor of shape `(N,)` indicating the first edge
                index for each example in the mesh
            max_segms: Scalar equal to maximum number of edges in the batch
        Returns:
            dists: FloatTensor of shape `(S,)`, where `dists[s]` is the squared
                euclidean distance of `s`-th edge to the closest point in the
                corresponding example in the batch
            idxs: LongTensor of shape `(S,)` indicating the closest point in the
                corresponding example in the batch.

            `dists[s] = d(points[idxs[s]], edges[s, 0], edges[s, 1])`,
            where `d(u, v0, v1)` is the distance of point `u` from the segment
            spanned by `(v0, v1)`.
        """
        dists, idxs = _C.edge_point_dist_forward(
            points, points_first_idx, segms, segms_first_idx, max_segms
        )
        ctx.save_for_backward(points, segms, idxs)
        return dists

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, segms, idxs = ctx.saved_tensors
        grad_points, grad_segms = _C.edge_point_dist_backward(
            points, segms, idxs, grad_dists
        )
        return grad_points, None, grad_segms, None, None


# pyre-fixme[16]: `_EdgePointDistance` has no attribute `apply`.
edge_point_distance = _EdgePointDistance.apply


def point_mesh_face_distance(meshes: Meshes, pcls: Pointclouds):
    """
    Computes the distance between a pointcloud and a mesh within a batch.
    Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
    sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`

    `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
        to the closest triangular face in mesh and averages across all points in pcl
    `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
        mesh to the closest point in pcl and averages across all faces in mesh.

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds

    Returns:
        loss: The `point_face(mesh, pcl) + face_point(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )
    return point_to_face
    # weight each example by the inverse of number of points in the example
    point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i),)
    num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
    weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
    weights_p = 1.0 / weights_p.float()
    point_to_face = point_to_face * weights_p
    point_dist = point_to_face.sum() / N
    return point_dist

    # face to point distance: shape (T,)
    face_to_point = face_point_distance(
        points, points_first_idx, tris, tris_first_idx, max_tris
    )

    # weight each example by the inverse of number of faces in the example
    tri_to_mesh_idx = meshes.faces_packed_to_mesh_idx()  # (sum(T_n),)
    num_tris_per_mesh = meshes.num_faces_per_mesh()  # (N, )
    weights_t = num_tris_per_mesh.gather(0, tri_to_mesh_idx)
    weights_t = 1.0 / weights_t.float()
    face_to_point = face_to_point * weights_t
    face_dist = face_to_point.sum() / N

    return point_dist + face_dist
    # return point_dist

def point_mesh_face_distance_per(meshes: Meshes, pcls: Pointclouds):
    """
    Computes the distance between a pointcloud and a mesh within a batch.
    Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
    sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`

    `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
        to the closest triangular face in mesh and averages across all points in pcl
    `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
        mesh to the closest point in pcl and averages across all faces in mesh.

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds

    Returns:
        loss: The `point_face(mesh, pcl) + face_point(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )
    return point_to_face
    # # weight each example by the inverse of number of points in the example
    # point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i),)
    # num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
    # weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
    # weights_p = 1.0 / weights_p.float()
    # point_to_face = point_to_face * weights_p
    # point_dist = point_to_face.sum() / N
    # return point_dist


def ICPLoss(mesh, pcl, faces):
    batch_size, point_num, _ = pcl.size()
    pytorch3d_pcl = Pointclouds(pcl)
    pytorch3d_mesh = Meshes(mesh, faces.unsqueeze(0).repeat(batch_size, 1, 1))
    per_point_dis = point_mesh_face_distance(pytorch3d_mesh, pytorch3d_pcl)
    per_point_dis = per_point_dis.view(batch_size, point_num)
    return per_point_dis.mean(-1)


def FingerICPLoss(mesh, pcl, faces, pcl_seg):
    batch_size, point_num, _ =pcl.size()
    pcl_finger = pcl.unsqueeze(1).repeat(1, 5, 1, 1).view(batch_size*5, point_num, 3)
    pytorch3d_pcl = Pointclouds(pcl_finger)
    mesh_finger = mesh.unsqueeze(1).repeat(1, 5, 1, 1).view(batch_size*5, -1, 3)
    mesh_finger = list(torch.chunk(mesh_finger, batch_size*5, dim=0))
    for index in range(batch_size*5):
        mesh_finger[index] = mesh_finger[index].squeeze(0)
    pytorch3d_mesh = Meshes(mesh_finger, faces*batch_size)
    per_point_dis = point_mesh_face_distance_per(pytorch3d_mesh, pytorch3d_pcl)
    per_point_dis = per_point_dis.view(batch_size, 5, point_num)
    finger_loss = []
    for index in range(5):
        dis = torch.where(pcl_seg.eq(index + 1), per_point_dis[:, index], torch.zeros_like(per_point_dis[:, index]))
        valid_num = dis.gt(0).sum(-1)
        loss = dis.sum(-1) / (valid_num + 1e-8)
        loss = torch.where(valid_num.eq(0), torch.zeros_like(loss), loss)
        finger_loss.append(loss)
    return torch.stack(finger_loss,dim=-1)


def JointICPLoss(mesh, pcl, faces, pcl_seg):
    batch_size, point_num, _ = pcl.size()
    pcl_joint = pcl.unsqueeze(1).repeat(1, 15, 1, 1).view(batch_size*15, point_num, 3)
    pytorch3d_pcl = Pointclouds(pcl_joint)
    mesh_finger = mesh.unsqueeze(1).repeat(1, 15, 1, 1).view(batch_size*15, -1, 3)
    mesh_finger = list(torch.chunk(mesh_finger, batch_size*15, dim=0))
    for index in range(batch_size*15):
        mesh_finger[index] = mesh_finger[index].squeeze(0)
    pytorch3d_mesh = Meshes(mesh_finger, faces*batch_size)
    per_point_dis = point_mesh_face_distance_per(pytorch3d_mesh, pytorch3d_pcl)
    per_point_dis = per_point_dis.view(batch_size, 15, point_num)
    finger_loss = []
    for index in range(15):
        dis = torch.where(pcl_seg.eq(index + 1), per_point_dis[:, index], torch.zeros_like(per_point_dis[:, index]))
        valid_num = dis.gt(0).sum(-1)
        loss = dis.sum(-1) / (valid_num + 1e-8)
        loss = torch.where(valid_num.eq(0), torch.zeros_like(loss), loss)
        finger_loss.append(loss)
    return torch.stack(finger_loss, dim=-1)