"""
Find matches between components
"""
import shutil
from pathlib import Path

import numpy as np

from filter_images_gps import images_from_model, get_metadata, images_from_dir
from utils.read_write_model import read_images_binary, read_cameras_binary, read_points3D_binary, write_points3D_binary

model_path = Path("/Users/dawars/3d_recon/hloc/sfm_aliked-lightglue_pairs-eigenplaces-50_0.1/snapshots/-1376481519")
out_model_path = Path(
    "/Users/dawars/3d_recon/hloc/sfm_aliked-lightglue_pairs-eigenplaces-50_0.1/snapshots/cathedral_matches")

image_names = images_from_model(model_path)

# budapest_images = []
# all_images = []
# for image in image_names:
#     metadata = get_metadata(Path("/Users/dawars/datasets/fortepan") / image)
#     if "orszag_name" in metadata and metadata["orszag_name"][0] == "Magyarország":
#         if "varos_name" in metadata and "budapest" in metadata["varos_name"][0].lower():
#             budapest_images.append(image)
#     all_images.append(image)
#
# foreign_images = set(all_images).difference(budapest_images)

# test
# budapest_images = set(["fortepan_32598.jpg"])
# foreign_images = set(["fortepan_82629.jpg"])
budapest_images = images_from_dir(
    Path("/Users/dawars/3d_recon/hloc/sfm_aliked-lightglue_pairs-eigenplaces-50_0.1/snapshots/cathedral_images"))
foreign_images = set(image_names).difference(budapest_images)

# foreign_images_dir = out_model_path.parent / "foreign_images"
# budapest_images_dir = out_model_path.parent / "budapest_images"
# foreign_images_dir.mkdir(exist_ok=True)
# budapest_images_dir.mkdir(exist_ok=True)

# for path in foreign_images:
#     if not Path(f"/Users/dawars/datasets/fortepan_preview/{path}").exists():
#         print(path)
#         continue
#     shutil.copy(f"/Users/dawars/datasets/fortepan_preview/{path}", foreign_images_dir)
#
# for path in budapest_images:
#     if not Path(f"/Users/dawars/datasets/fortepan_preview/{path}").exists():
#         print(path)
#         continue
#     shutil.copy(f"/Users/dawars/datasets/fortepan_preview/{path}", budapest_images_dir)

images = read_images_binary(model_path / "images.bin")
points = read_points3D_binary(model_path / "points3D.bin")
cameras = read_cameras_binary(model_path / "cameras.bin")

image_to_id = {img.name: int(img.id) for img in images.values()}
id_to_image = {int(img.id): img.name for img in images.values()}

foreign_im_matches = []
bp_im_matches = []
saved_pts = {}
for point in points.values():
    image_names = [id_to_image[pt] for pt in point.image_ids]
    if set(budapest_images).intersection(image_names) and \
            foreign_images.intersection(image_names):
        saved_pts[point.id] = point
        bp_im_matches.append(len(set(budapest_images).intersection(image_names)))
        foreign_im_matches.append(len(set(foreign_images).intersection(image_names)))

print(np.max(bp_im_matches))
print(np.max(foreign_im_matches))
print(np.argmax(bp_im_matches))
print(np.argmax(foreign_im_matches))

# write_images_binary(out_model_path / "images.bin")
write_points3D_binary(saved_pts, out_model_path / "points3D.bin")
# write_cameras_binary(out_model_path / "cameras.bin")

print(points)
