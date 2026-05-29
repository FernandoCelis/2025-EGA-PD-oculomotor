import torch
import torch.utils.data as data
from torchvision.transforms import transforms

from PIL import Image
import os
import random
import logging


class EyeVideoDataset(data.Dataset):
    def __init__(self, root_dir: str, names_patients: list, data_augmentation: bool, subsampling_step: int = 1, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.subsampling_step = subsampling_step
        self.video_folders = [(folder, "augmNone") for folder in os.listdir(root_dir) if folder[:3] in names_patients]

        logging.info(f"Number of patient names: {len(self.video_folders)}")
        if data_augmentation == True:
            logging.info("Using Data augmentation. Augmenting with Horizontal Flip")
            self.video_folders.extend((folder, "augm1") for folder in os.listdir(root_dir) if folder[:3] in names_patients)
            self.augm1 = transforms.RandomHorizontalFlip(p=1)

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, index):
        name_patient, augm_info = self.video_folders[index]
        folder_path = os.path.join(self.root_dir, name_patient)
        frames = []

        for i, frame_name in enumerate(sorted(os.listdir(folder_path))):
            if i % self.subsampling_step == 0:
                frame_path = os.path.join(folder_path, frame_name)
                frame = Image.open(frame_path)
                if self.transform is not None:
                    frame = self.transform(frame)
                    if augm_info == "augm1":
                        frame = self.augm1(frame)
                frames.append(frame)

        video_tensor = torch.stack(frames).transpose(0, 1)
        label = int(name_patient[0] == 'P')
        name_patient = name_patient + "_" + augm_info
        return video_tensor, label, name_patient


def get_5kf_fold_partitions(seed):
    parkinson_patients = [f"P{index:02}" for index in range(25)]
    control_patients = [f"C{index:02}" for index in range(25)]
    all_patients = parkinson_patients + control_patients
    available_patients = all_patients.copy()

    folds_dict = {}
    for i in range(1, 6):
        val_set = get_balanced_5fold_subset(available_patients, random_seed=seed)
        folds_dict[f"fold_{i}_val"] = val_set
        folds_dict[f"fold_{i}_train"] = [p for p in all_patients if p not in val_set]
        available_patients = [p for p in available_patients if p not in val_set]
        seed += 1

    return folds_dict


def get_balanced_5fold_subset(input_list, random_seed):
    subset_found = False
    while not subset_found:
        c_count, p_count = 0, 0
        subset = []
        for element in get_unique_random_subset(input_list=input_list, num_elements=10, random_seed=random_seed):
            if "C" in element:
                c_count += 1
            else:
                p_count += 1
            subset.append(element)
        if c_count >= 5 and p_count >= 5:
            subset_found = True
        else:
            random_seed += 1
    return subset


def get_unique_random_subset(input_list, num_elements, random_seed=None):
    subset_found = False
    while not subset_found:
        if random_seed is not None:
            random.seed(random_seed)
            random_subset = sorted(random.sample(input_list, num_elements))
        else:
            random_subset = sorted(random.sample(input_list, num_elements))
        if len(set(random_subset)) == num_elements:
            subset_found = True
    return random_subset


class LateralVideoDataset(data.Dataset):
    def __init__(self, root_dir: str, names_patients: list, data_augmentation: bool, subsampling_step: int = 1, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.subsampling_step = subsampling_step

        self.patients = []
        for patient_id in names_patients:
            left_folder = os.path.join(root_dir, f"{patient_id}_left")
            right_folder = os.path.join(root_dir, f"{patient_id}_right")
            if os.path.isdir(left_folder) and os.path.isdir(right_folder):
                self.patients.append((patient_id, "augmNone"))
            else:
                logging.warning(f"Missing left/right folder for patient {patient_id}, skipping.")

        logging.info(f"Number of patients with both eyes: {len(self.patients)}")

        if data_augmentation:
            logging.info("Using Data augmentation. Augmenting with Horizontal Flip")
            originals = [(p, "augmNone") for p, _ in self.patients]
            self.patients.extend([(p, "augm1") for p, _ in originals])
            self.augm1 = transforms.RandomHorizontalFlip(p=1)

    def __len__(self):
        return len(self.patients)

    def _load_frames(self, folder_path: str, augm_info: str) -> torch.Tensor:
        frames = []
        for i, frame_name in enumerate(sorted(os.listdir(folder_path))):
            if i % self.subsampling_step == 0:
                frame = Image.open(os.path.join(folder_path, frame_name))
                if self.transform is not None:
                    frame = self.transform(frame)
                    if augm_info == "augm1":
                        frame = self.augm1(frame)
                frames.append(frame)
        return torch.stack(frames).transpose(0, 1)

    def __getitem__(self, index):
        patient_id, augm_info = self.patients[index]

        left_tensor  = self._load_frames(os.path.join(self.root_dir, f"{patient_id}_left"),  augm_info)
        right_tensor = self._load_frames(os.path.join(self.root_dir, f"{patient_id}_right"), augm_info)

        label = int(patient_id[0] == 'P')
        name_patient = f"{patient_id}_LR_{augm_info}"

        return (left_tensor, right_tensor), label, name_patient
