import numpy as np
import torchvision.datasets as datasets

DATASET_INFO = {
    "cifar10": {
        "num_classes": 10,
        "input_shape": (3, 32, 32),
        "model": "cnn_cifar10",
        "output_dir": "output-cifar-10",
    },
    "mnist": {
        "num_classes": 10,
        "input_shape": (1, 28, 28),
        "model": "cnn_mnist",
        "output_dir": "output-mnist",
    },
    "fashion_mnist": {
        "num_classes": 10,
        "input_shape": (1, 28, 28),
        "model": "cnn_fashion_mnist",
        "output_dir": "output-fashion-mnist",
    },
    "gtsrb": {
        "num_classes": 43,
        "input_shape": (3, 32, 32),
        "model": "cnn_gtsrb",
        "output_dir": "output-gtsrb",
    },
}

VALID_DATASETS = list(DATASET_INFO.keys())


def get_dataset_info(dataset_name: str, output_dir: str | None = None) -> dict:
    if dataset_name not in DATASET_INFO:
        raise ValueError(
            f"Dataset inválido '{dataset_name}'. Opções: {VALID_DATASETS}"
        )
    info = DATASET_INFO[dataset_name].copy()
    if output_dir is not None:
        info["output_dir"] = str(output_dir)
    return info


def load_dataset(dataset_name: str):
    if dataset_name not in DATASET_INFO:
        raise ValueError(
            f"Dataset inválido '{dataset_name}'. Opções: {VALID_DATASETS}"
        )
    loaders = {
        "cifar10": _load_cifar10,
        "mnist": _load_mnist,
        "fashion_mnist": _load_fashion_mnist,
        "gtsrb": _load_gtsrb,
    }
    return loaders[dataset_name]()


def _load_cifar10():
    train = datasets.CIFAR10(root="./data", train=True, download=True)
    test = datasets.CIFAR10(root="./data", train=False, download=True)
    x_train = np.array(train.data, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
    y_train = np.array(train.targets, dtype=np.int64)
    x_test = np.array(test.data, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
    y_test = np.array(test.targets, dtype=np.int64)
    return (x_train, y_train), (x_test, y_test)


def _load_mnist():
    train = datasets.MNIST(root="./data", train=True, download=True)
    test = datasets.MNIST(root="./data", train=False, download=True)
    x_train = train.data.numpy().astype(np.float32).reshape(-1, 1, 28, 28) / 255.0
    y_train = train.targets.numpy().astype(np.int64)
    x_test = test.data.numpy().astype(np.float32).reshape(-1, 1, 28, 28) / 255.0
    y_test = test.targets.numpy().astype(np.int64)
    return (x_train, y_train), (x_test, y_test)


def _load_fashion_mnist():
    train = datasets.FashionMNIST(root="./data", train=True, download=True)
    test = datasets.FashionMNIST(root="./data", train=False, download=True)
    x_train = train.data.numpy().astype(np.float32).reshape(-1, 1, 28, 28) / 255.0
    y_train = train.targets.numpy().astype(np.int64)
    x_test = test.data.numpy().astype(np.float32).reshape(-1, 1, 28, 28) / 255.0
    y_test = test.targets.numpy().astype(np.int64)
    return (x_train, y_train), (x_test, y_test)


def _load_gtsrb():
    import os
    import urllib.request
    import zipfile

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "gtsrb")
    train_dir = os.path.join(data_dir, "GTSRB", "Final_Training", "Images")
    test_dir = os.path.join(data_dir, "GTSRB", "Final_Test", "Images")

    train_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
    test_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"
    test_gt_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"

    os.makedirs(data_dir, exist_ok=True)

    def download_and_extract(url, dest):
        zip_path = os.path.join(data_dir, os.path.basename(url))
        if not os.path.exists(zip_path):
            print(f"Baixando {os.path.basename(url)}...")
            urllib.request.urlretrieve(url, zip_path)
        print(f"Extraindo {os.path.basename(url)}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest)

    if not os.path.exists(train_dir):
        download_and_extract(train_url, data_dir)
    if not os.path.exists(test_dir):
        download_and_extract(test_url, data_dir)
    if not os.path.exists(os.path.join(data_dir, "GT-final_test.csv")):
        download_and_extract(test_gt_url, data_dir)

    import csv
    from PIL import Image

    def load_ppm_image(path):
        img = Image.open(path).resize((32, 32))
        return np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0

    train_images, train_labels = [], []
    for class_id in range(43):
        class_dir = os.path.join(train_dir, f"{class_id:05d}")
        csv_file = os.path.join(class_dir, f"GT-{class_id:05d}.csv")
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                img_path = os.path.join(class_dir, row["Filename"])
                train_images.append(load_ppm_image(img_path))
                train_labels.append(class_id)

    test_csv = os.path.join(data_dir, "GT-final_test.csv")
    test_images, test_labels = [], []
    with open(test_csv, "r") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            img_path = os.path.join(test_dir, row["Filename"])
            test_images.append(load_ppm_image(img_path))
            test_labels.append(int(row["ClassId"]))

    x_train = np.array(train_images)
    y_train = np.array(train_labels, dtype=np.int64)
    x_test = np.array(test_images)
    y_test = np.array(test_labels, dtype=np.int64)

    return (x_train, y_train), (x_test, y_test)
