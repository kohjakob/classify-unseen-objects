
from pipeline_tasks.instances_feature_loading import load_instance_features_from_npz
from pipeline_conf.conf import PATHS

if __name__ == "__main__":
    print("TEST!")
    folder = PATHS.scannet_gt_instance_output_dir
    features_dict = load_instance_features_from_npz(folder)

    print("Anzahl geladener Instanzen:", len(features_dict))
    for name, features in features_dict.items():
        print(f"{name}: {features.shape}")

    # PYTHONPATH=. python pipeline_tasks/test_file.py

    