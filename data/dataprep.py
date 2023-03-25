from pathlib import Path
import numpy as np
import nibabel as nib


def get_cmr_niis(d, mod):
    """
    recursively get paths of nii files of a given modality(mod) from a data path (d)
    """
    return sorted((p for p in d.rglob(f'*{mod}.nii*') if not p.name.startswith('.')), key=lambda x: x.name)


def get_label_nii(d, mod):
    """
    recursively get paths of gt_label nii files of a given modality(mod) from a data path (d)
    """
    return sorted((p for p in d.rglob(f'*{mod}_manual.nii*') if not p.name.startswith('.')), key=lambda x: x.name)


def train_valid_test(total_n, train_n, valid_n):
    """return index for train, validation, and test"""
    np.random.seed(233)
    all_indices = np.arange(total_n)

    # indices for training
    train_indices = np.random.choice(all_indices, size=train_n, replace=False)

    # indices for validation
    remaining_indices = np.setdiff1d(all_indices, train_indices)
    valid_indices = np.random.choice(remaining_indices, size=valid_n, replace=False)

    # indices for testing
    test_n = total_n - train_n - valid_n
    remaining_indices = np.setdiff1d(remaining_indices, valid_indices)
    test_indices = np.random.choice(remaining_indices, size=test_n, replace=False)

    return train_indices, valid_indices, test_indices


if __name__ == '__main__':
    # directory for images and labels
    DIR = '../../Data/MSCMR'
    DIR = Path(DIR)

    # image and label paths for DE
    image_de_paths = get_cmr_niis(DIR, 'DE')
    label_de_paths = get_label_nii(DIR, 'DE')
    assert len(image_de_paths) == len(label_de_paths)

    # random split 25 train, 5 valid, 15 test
    train_indices, valid_indices, test_indices = train_valid_test(len(image_de_paths), 25, 5)

    # training data
    # for each line: image_path, label_path, slice_idx
    with open('../../Data/MSCMR/train.txt','w') as f:
        for idx in train_indices:
            image_path = image_de_paths[idx]
            n_slices = nib.load(image_path).shape[2]
            for slice_idx in range(n_slices):
                f.write(f'{image_path.relative_to(DIR)}\t{label_de_paths[idx].relative_to(DIR)}\t{slice_idx}\n')

    # validation data
    # for each line: image_path, label_path, slice_idx
    with open('../../Data/MSCMR/valid.txt','w') as f:
        for idx in valid_indices:
            image_path = image_de_paths[idx]
            n_slices = nib.load(image_path).shape[2]
            for slice_idx in range(n_slices):
                f.write(f'{image_path.relative_to(DIR)}\t{label_de_paths[idx].relative_to(DIR)}\t{slice_idx}\n')

    # testng data
    # for each line: image_path, dx, dy, dz
    with open('../../Data/MSCMR/test.txt','w') as f:
        for idx in test_indices:
            image_path = image_de_paths[idx]
            dx, dy, dz = nib.load(image_path).shape
            f.write(f'{image_path.relative_to(DIR)}\t{dx}\t{dy}\t{dz}\n')