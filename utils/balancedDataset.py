import os
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple

from torchvision.datasets.folder import ImageFolder, DatasetFolder, default_loader, IMG_EXTENSIONS



class BalancedDataset(ImageFolder):
    """A data loader that implements slices for ImageFolder.

    Args:
        slices (list, optional): A list of slices used to keep only a certain number of samples.
            For the class with index i the slice at index i % len(slices) is used.
        check_images (boolean, default True): If True, check if there are files that cannot be
            read by PIL. The creation of the dataset will be slower.
    """

    def is_valid_image(self, path: str) -> bool:
        if not os.path.exists(path):
            return False

        img = None
        try:
            img = default_loader(path)
        except:
            return False

        return img is not None

    def filter_loader(self, path: str) -> Any:
        target = default_loader(path)

        if self.filter:
            return self.filter(target)

        return target

    def make_dataset(
        self,
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        If specified, it cuts the dataset using balancing passed to the constructor.
        """

        directory = os.path.expanduser(directory)

        instances = []

        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue

            class_instances = []

            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)

                    if extensions:
                        _, fileextension = os.path.splitext(path)
                        if not fileextension.lower() in extensions:
                            print("[🛑 ERROR] Found non valid extension:", path)
                            continue

                    if is_valid_file and not is_valid_file(path):
                        print("[🛑 ERROR] Found non valid sample:", path)
                        continue

                    item = path, class_index
                    class_instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

            num_samples = len(class_instances)

            if self.balance:
                try:
                    samples_balance = int(
                        num_samples * (self.balance[class_index] / np.max(self.balance)))

                    class_instances = class_instances[:samples_balance]
                except:
                    raise ValueError("[🛑 ERROR] Invalid balance")

            if self.datasetSize:
                try:
                    class_instances = class_instances[:self.datasetSize]
                except:
                    raise ValueError("Invalid dataset size")

            instances.extend(class_instances)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"[🛑 ERROR] Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"[🛑 ERROR] Supported extensions are: {', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances

    def __init__(self, root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 balance: Optional[List[int]] = None,
                 datasetSize: Optional[int] = None,
                 check_images: Optional[bool] = True,
                 use_cache: Optional[bool] = False,
                 transform_filter: Optional[Callable] = None,
                 with_path: Optional[bool] = False):

        self.balance = balance
        self.use_cache = use_cache
        self.cached_data = {}
        self.filter = transform_filter
        self.datasetSize = datasetSize
        self.with_path = with_path

        super(DatasetFolder, self).__init__(
            root, transform=transform, target_transform=target_transform)

        classes, class_to_idx = self._find_classes(
            self.root) if hasattr(self, "_find_classes") else self.find_classes(self.root)

        samples = self.make_dataset(self.root,
                                    class_to_idx,
                                    IMG_EXTENSIONS if check_images else None,
                                    self.is_valid_image if check_images else None)

        self.loader = self.filter_loader
        self.extensions = IMG_EXTENSIONS if check_images else None

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.imgs = self.samples

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, _ = self.samples[index]

        if self.use_cache and path in self.cached_data.keys():
            return self.cached_data[path]

        sample, target = super().__getitem__(index)

        if self.use_cache:
            self.cached_data[path] = (sample, target)

        if self.with_path:
            return sample, target, path

        return sample, target