from typing import Optional, List, Dict, Any
import multiprocessing
from pathlib import Path
import pycolmap
import subprocess
import os
import os.path as osp

from src.sfm_runner.utils.make_database import load_intrin_to_database
from . import logger
from .utils.database import COLMAPDatabase
from .triangulation import (
    import_features, import_matches, estimation_and_geometric_verification,
    OutputCapture, NOT_EXPO_COLMAP_CFGS)
COLMAP_PATH = os.environ.get("COLMAP_PATH", 'colmap') # 'colmap is default value

def create_empty_db(database_path: Path):
    if database_path.exists():
        logger.warning('The database already exists, deleting it.')
        database_path.unlink()
    logger.info('Creating an empty database...')
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()


def import_images(image_dir: Path,
                  database_path: Path,
                  camera_mode: pycolmap.CameraMode,
                  image_list: Optional[List[str]] = None,
                  options: Optional[Dict[str, Any]] = None):
    logger.info(f'Importing images into the database, camera mode is {camera_mode}...')
    if options is None:
        options = {}
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f'No images found in {image_dir}.')
    with pycolmap.ostream():
        pycolmap.import_images(database_path, image_dir, camera_mode,
                               image_list=image_list or [],
                               options=options)


def get_image_ids(database_path: Path) -> Dict[str, int]:
    db = COLMAPDatabase.connect(database_path)
    images = {}
    for name, image_id in db.execute("SELECT name, image_id FROM images;"):
        images[name] = image_id
    db.close()
    return images

def run_reconstruction(sfm_dir, database_path, image_dir, colmap_configs=None, verbose=False):
    models_path = sfm_dir / 'models'

    models_path.mkdir(exist_ok=True, parents=True)
    if colmap_configs['colmap_mapper_cfgs'] is None:
        logger.info(f"Use PyCOLMAP for reconstruction...")
        mapper_options = pycolmap.IncrementalMapperOptions(ba_global_use_pba=colmap_configs['use_pba'], ba_refine_focal_length=not colmap_configs['no_refine_intrinsics'], ba_refine_extra_params=not colmap_configs['no_refine_intrinsics'], num_threads=min(multiprocessing.cpu_count(), colmap_configs['n_threads'] if 'n_threads' in colmap_configs else 16))
        logger.info('Running 3D reconstruction...')
        with OutputCapture(verbose):
            with pycolmap.ostream():
                logger.info(f"use: {min(multiprocessing.cpu_count(), colmap_configs['n_threads'] if 'n_threads' in colmap_configs else 16)} cpus")
                logger.info(mapper_options.summary())
                reconstructions = pycolmap.incremental_mapping(
                    database_path, image_dir, models_path,
                    mapper_options,)
    else:
        logger.info(f"Use command line COLMAP for reconstruction...")
        cmd = [COLMAP_PATH, "mapper"]
        cmd += ["--image_path", str(image_dir)]
        cmd += ["--database_path", str(database_path)]
        cmd += ["--output_path", str(models_path)]
        if colmap_configs is not None and "min_model_size" in colmap_configs:
            cmd += ["--Mapper.min_model_size", str(colmap_configs["min_model_size"])]
        cmd += ["--Mapper.num_threads", str(min(multiprocessing.cpu_count(), colmap_configs['n_threads'] if 'n_threads' in colmap_configs else 16))]

        if colmap_configs['use_pba']:
            cmd += ["--Mapper.ba_global_use_pba", '1']

        if colmap_configs['colmap_mapper_cfgs'] is not None:
            for config_name, value in colmap_configs["colmap_mapper_cfgs"].items():
                if config_name in NOT_EXPO_COLMAP_CFGS:
                    cmd += [NOT_EXPO_COLMAP_CFGS[config_name], str(value)]
        
        if (
            colmap_configs is not None
            and colmap_configs["no_refine_intrinsics"] is True
        ):
            cmd += [
                "--Mapper.ba_refine_focal_length",
                "0",
                "--Mapper.ba_refine_extra_params",
                "0",
            ]

        if verbose:
            logger.info(' '.join(cmd))
            colmap_res = subprocess.run(cmd)
        else:
            colmap_res = subprocess.run(cmd, capture_output=True)
            with open(osp.join(models_path, "output.txt"), "w") as f:
                f.write(colmap_res.stdout.decode())

        reconstructions = {}
        for id, model_path in enumerate(sorted(models_path.glob('*'))):
            if model_path.is_dir():
                reconstructions[id] = pycolmap.Reconstruction(model_path)

    if len(reconstructions) == 0:
        logger.error('Could not reconstruct any model!')
        os.system(f"mv {models_path}/* {sfm_dir}")
        os.system(f"rm -rf {models_path}")
        return None
    logger.info(f'Reconstructed {len(reconstructions)} model(s).') if verbose else None

    largest_index = None
    largest_num_images = 0
    for index, rec in reconstructions.items():
        num_images = rec.num_reg_images()
        if num_images > largest_num_images:
            largest_index = index
            largest_num_images = num_images
    assert largest_index is not None
    logger.info(f'Largest model is #{largest_index} '
                f'with {largest_num_images} images.') if verbose else None

    os.system(f"mv {models_path}/* {sfm_dir}")
    os.system(f"rm -rf {models_path}")
    return reconstructions[largest_index]


def main(sfm_dir, image_dir, pairs, features, matches, prior_intrin,
         camera_mode=pycolmap.CameraMode.AUTO, verbose=False,
         skip_geometric_verification=False, min_match_score=None,
         image_list: Optional[List[str]] = None, colmap_configs = None):

    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / 'database.db'

    create_empty_db(database)
    if colmap_configs['use_pba'] or colmap_configs["ImageReader_camera_mode"] == 'per_image':
        camera_mode = pycolmap.CameraMode.PER_IMAGE
    elif colmap_configs["ImageReader_camera_mode"] == 'single_camera':
        camera_mode = pycolmap.CameraMode.SINGLE
    
    camera_model = "SIMPLE_RADIAL" if 'ImageReader_camera_model' not in colmap_configs else colmap_configs['ImageReader_camera_model']
    if colmap_configs['use_pba']:
        camera_model = "SIMPLE_RADIAL"

    img_import_opts = pycolmap.ImageReaderOptions(camera_model=camera_model)
    import_images(image_dir, database, camera_mode, image_list, img_import_opts)
    
    if prior_intrin is not None:
        logger.info(f"Load prior intrin into db...")
        if colmap_configs['use_pba']:
            logger.warning('Currently PBA not support fix (known) intrin and optimize poses and point clouds. Moreover, the loaded PINHOLE camera model is not supported.\n PBA is disabled automatically.')
            colmap_configs['use_pba'] = False
        if colmap_configs['ImageReader_camera_mode'] == "per_image":
            logger.warning('Currently PBA !')
            colmap_configs['use_pba'] = True
        load_intrin_to_database(database, prior_intrin, colmap_configs)

    image_ids = get_image_ids(database)
    import_features(image_ids, database, features, verbose=verbose)
    import_matches(image_ids, database, pairs, matches,
                   min_match_score, skip_geometric_verification, verbose=verbose)
    if not skip_geometric_verification:
        max_error = 4.0 if 'geometry_verify_thr' not in colmap_configs else colmap_configs['geometry_verify_thr']
        estimation_and_geometric_verification(database, pairs, verbose, max_error=max_error)

    reconstruction = run_reconstruction(sfm_dir, database, image_dir, colmap_configs=colmap_configs, verbose=verbose)
    if reconstruction is not None and verbose:
        logger.info(f'Reconstruction statistics:\n{reconstruction.summary()}'
                    + f'\n\tnum_input_images = {len(image_ids)}')
    return reconstruction