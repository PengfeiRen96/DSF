import json
import os
import traceback
import pickle

from joblib import Parallel, delayed
import numpy as np
import trimesh
from matplotlib import pyplot as plt

closed_hand = "assets/mano/closed_mano.obj"
_, hand_faces = load_obj(closed_hand, normalization=False)


def load_obj(filename_obj, normalization=True):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """

    # load vertices
    vertices = []
    for line in open(filename_obj).readlines():
        if len(line.split()) == 0:
            continue
        if line.split()[0] == "v":
            vertices.append([float(v) for v in line.split()[1:4]])
    vertices = np.vstack(vertices).astype("float32")

    # load faces
    faces = []
    for line in open(filename_obj).readlines():
        if len(line.split()) == 0:
            continue
        if line.split()[0] == "f":
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split("/")[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split("/")[0])
                v2 = int(vs[i + 2].split("/")[0])
                faces.append((v0, v1, v2))
    faces = np.vstack(faces).astype("int32") - 1

    # normalize into a unit cube centered zero
    if normalization:
        vertices -= vertices.min(0)[None, :]
        vertices /= np.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0)[None, :] / 2

    return vertices, faces


def load_batch_info(save_path, faces_right, faces_left, scale=0.001, get_depth=True):
    with open(save_path, "rb") as p_f:
        batch_data = pickle.load(p_f)
    sample, results = batch_data["sample"], batch_data["results"]
    obj_faces = results["objfaces"]
    hand_verts = results["verts"]
    obj_verts = results["objpoints3d"]
    penetr_masks = results["contact_info"]["repulsion_masks"]

    hand_faces = []
    for side in sample["sides"]:
        if side == "right":
            hand_faces.append(faces_right)
        else:
            hand_faces.append(faces_left)
    hand_faces = np.stack(hand_faces)
    sample_infos = []
    for hand_vert, hand_face, obj_vert, penetr_mask in zip(
        hand_verts, hand_faces, obj_verts, penetr_masks
    ):
        obj_mesh = trimesh.load({"vertices": obj_vert, "faces": obj_faces})
        trimesh.repair.fix_normals(obj_mesh)

        sample_info = {
            "hand_verts": hand_vert * scale,
            "hand_faces": hand_face,
            "obj_verts": np.array(obj_mesh.vertices) * scale,
            "obj_faces": np.array(obj_mesh.faces),
        }
        if get_depth:
            if penetr_mask.sum() == 0:
                max_depth = 0
            else:
                (
                    result_close,
                    result_distance,
                    _,
                ) = trimesh.proximity.closest_point(
                    obj_mesh, hand_vert[penetr_mask == 1]
                )
                max_depth = result_distance.max()
            sample_info["max_depth"] = max_depth
        sample_infos.append(sample_info)

    return sample_infos


def intersect_vox(obj_mesh, hand_mesh, pitch=0.01):
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume


def intersect(obj_mesh, hand_mesh, engine="auto"):
    trimesh.repair.fix_normals(obj_mesh)
    inter_mesh = obj_mesh.intersection(hand_mesh, engine=engine)
    return inter_mesh


def get_all_volumes(exp_id, batch_step=1, workers=8):
    save_pickles = sorted(
        [
            os.path.join(exp_id, filename)
            for filename in os.listdir(exp_id)
            if ".pkl" in filename
        ]
    )
    batch_infos = Parallel(n_jobs=workers, verbose=5)(
        delayed(load_batch_info)(
            save_pickle, faces_right=hand_faces, faces_left=hand_faces
        )
        for save_pickle in save_pickles[::batch_step]
    )
    # Prepare simulation storing results
    sample_infos = [
        sample_info for batch_info in batch_infos for sample_info in batch_info
    ]
    volumes = get_volumes_from_samples(sample_infos, workers=workers)
    volumes_clean = [volume for volume in volumes if volume is not None]
    skipped = len(volumes) - len(volumes_clean)
    simulation_results_path = os.path.join(
        exp_id.replace("save_results", "simulation_results"),
        "results_volume_voxels_0_005.json",
    )
    with open(simulation_results_path, "w") as j_f:
        json.dump(
            {
                "mean_volume": np.mean(volumes_clean),
                "volumes": volumes_clean,
                "median_volume": np.median(volumes_clean),
                "std_volume": np.std(volumes_clean),
                "min_volume": np.min(volumes_clean),
                "max_volume": np.max(volumes_clean),
                "skipped": skipped,
                "computed": len(volumes_clean),
            },
            j_f,
        )
        print("Skipped {}, kept {}".format(skipped, len(volumes_clean)))


def get_volumes_from_samples(sample_infos, workers=8):
    volumes = Parallel(n_jobs=workers, verbose=5)(
        delayed(get_sample_intersect_volume)(sample_info)
        for sample_info in sample_infos
    )
    return volumes


def get_sample_intersect_volume(sample_info, mode="voxels"):
    hand_mesh = trimesh.Trimesh(
        vertices=sample_info["hand_verts"], faces=sample_info["hand_faces"]
    )
    obj_mesh = trimesh.Trimesh(
        vertices=sample_info["obj_verts"], faces=sample_info["obj_faces"]
    )
    if mode == "engines":
        try:
            intersection = intersect(obj_mesh, hand_mesh, engine="scad")
            if intersection.is_watertight:
                volume = intersection.volume
            else:
                intersection = intersect(obj_mesh, hand_mesh, engine="blender")
                # traceback.print_exc()
                if intersection.vertices.shape[0] == 0:
                    volume = 0
                elif intersection.is_watertight:
                    volume = intersection.volume
                else:
                    volume = None
        except Exception:
            intersection = intersect(obj_mesh, hand_mesh, engine="blender")
            # traceback.print_exc()
            if intersection.vertices.shape[0] == 0:
                volume = 0
            elif intersection.is_watertight:
                volume = intersection.volume
            else:
                volume = None
    elif mode == "voxels":
        volume = intersect_vox(obj_mesh, hand_mesh, pitch=0.005)
    return volume