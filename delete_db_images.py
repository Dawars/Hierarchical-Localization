import itertools
import shutil
from pathlib import Path

from filter_images_gps import images_from_list, images_from_model, images_from_dir
from hloc.utils.database import COLMAPDatabase, image_ids_to_pair_id


def delete_matches(db_path, image_names: str | list[str]):
    db = COLMAPDatabase.connect(db_path)

    if not isinstance(image_names, list):
        image_names = [image_names]

    for img1, img2 in image_names:
        # Fetch image_id for the given image_name
        image_id1_rows = db.execute("SELECT image_id FROM images WHERE name=?", (img1,))
        image_id1 = [row[0] for row in image_id1_rows]
        image_id2_rows = db.execute("SELECT image_id FROM images WHERE name=?", (img2,))
        image_id2 = [row[0] for row in image_id2_rows]

        # Delete matches associated with the image
        # pair_ids_to_delete = db.execute("SELECT pair_id FROM matches").fetchall()
        # for pair_id in pair_ids_to_delete:
        #     if image_ids[0] in pair_id_to_image_ids(pair_id[0]):
        if len(image_id1) == 0 or len(image_id2) == 0:
            continue
        pair_id = image_ids_to_pair_id(image_id1[0], image_id2[0])
        db.execute("DELETE FROM matches WHERE pair_id=?", (pair_id,))

        # Delete two-view geometries associated with the image
        db.execute("DELETE FROM two_view_geometries WHERE pair_id=?", (pair_id,))

    db.commit()

    db.close()


if __name__ == '__main__':
    # theater
    left_images = images_from_list(Path("/Users/dawars/3d_recon/theater_2024/features/image_list_left.txt"))
    right_images = images_from_list(Path("/Users/dawars/3d_recon/theater_2024/features/image_list_right.txt"))
    theater_db = "/Users/dawars/3d_recon/theater_2024/sfm_mast3r_aliked_unlimited_nms2/database.db"
    delete_matches(theater_db,
                  list(itertools.product(left_images, right_images)))

    #chain bridge
    buda_images = images_from_dir(Path("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-cosplace-50/chain_bridge/buda"))
    pest_images = images_from_dir(Path("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-cosplace-50/chain_bridge/pest"))
    north_images = images_from_dir(Path("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-cosplace-50/chain_bridge/north"))
    south_images = images_from_dir(Path("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-cosplace-50/chain_bridge/south"))
    matches_to_remove = (set(itertools.product(buda_images, pest_images))
                         .difference(itertools.product(north_images, north_images))
                         .difference(itertools.product(south_images, south_images)))
    # 100421 fp_219486
    # 8461 fp_108351

    Path("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-cosplace-50/chain_bridge/image_list_buda_chain_bridge.txt").write_text(
        "\n".join(buda_images))
    Path("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-cosplace-50/chain_bridge/image_list_pest_chain_bridge.txt").write_text(
        "\n".join(pest_images))
    Path("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-cosplace-50/chain_bridge/image_list_north_chain_bridge.txt").write_text(
        "\n".join(north_images))
    Path("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-cosplace-50/chain_bridge/image_list_south_chain_bridge.txt").write_text(
        "\n".join(south_images))


    delete_matches("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-cosplace-50/database.db.bridge",
                  list(matches_to_remove))

    chain_bridge = set(buda_images + pest_images)
    Path("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-cosplace-50/chain_bridge/image_list.txt").write_text(
        "\n".join(chain_bridge))

    # theater_images = set(images_from_list(Path("/Users/dawars/datasets/image_list_theater.txt")) + list(
    #     set(images_from_dir(Path("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-cosplace-50/kuria_theater/images")))
    #         - set(images_from_dir(Path("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-cosplace-50/kuria_theater/kuria")))
    # ))
    parlament_images = set(images_from_list(Path("/Users/dawars/datasets/image_list_parlament.txt")) +
                           images_from_list(Path("/Users/dawars/datasets/image_list_orszaghaz.txt")) +
                           images_from_list(Path("/Users/dawars/datasets/image_list_kuria.txt")) +
                           images_from_dir(Path(
                               "/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-cosplace-50/kuria_theater/kuria"))
                           )

    buda_castle_images = set(images_from_dir(Path("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-cosplace-50/buda_castle/buda_castle")) +
                             images_from_dir(Path("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-cosplace-50/chain_bridge/pest")))
    bazilika_images = set(images_from_list(Path("/Users/dawars/datasets/image_list_bazilika.txt")) +
                          images_from_list(Path("/Users/dawars/datasets/image_list_szt_istvan1.txt")) +
                          images_from_list(Path("/Users/dawars/datasets/image_list_szt_istvan2.txt")))
    kuria_images = set(images_from_dir(Path("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-cosplace-50/kuria_theater/kuria")))

    # todo delete matches between different locations
    matches_to_remove = list(itertools.product(bazilika_images, parlament_images))

    # delete_matches("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-cosplace-50/database.db",
    #                matches_to_remove)
    #
    # delete_matches("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-cosplace-50/database.db",
    #                list(itertools.product(pest_images, buda_images)))
