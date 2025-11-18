# ...existing code...
import os
import numpy as np
from random import shuffle
from PIL import Image
from .bucket_utils import load_images_from_bucket

def _load_images_from_dir(dir_path, label, img_size):
    imgs = []
    if not os.path.isdir(dir_path):
        return imgs
    for fname in os.listdir(dir_path):
        fpath = os.path.join(dir_path, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            img = Image.open(fpath).convert("RGB").resize((img_size, img_size))
            imgs.append((np.array(img), label))
        except Exception:
            continue
    return imgs

def load_data(s3_client=None, bucket_name=None, img_size=224, local_base="data/xray"):
    """
    Carrega dados em ordem de preferência:
    1) Diretório local (ex.: após `dvc pull` -> data/xray/NORMAL e data/xray/PNEUMONIA)
    2) Bucket S3/MinIO via load_images_from_bucket

    Args:
        s3_client: boto3 client (opcional, usado se não houver dados locais)
        bucket_name: nome do bucket (opcional)
        img_size: tamanho da imagem (quadrada)
        local_base: base local onde DVC coloca os dados

    Retorna:
        X (np.array), y (np.array)
    """
    data = []
    normal_dir = os.path.join(local_base, "NORMAL")
    pneumonia_dir = os.path.join(local_base, "PNEUMONIA")

    if os.path.isdir(normal_dir) and os.path.isdir(pneumonia_dir):
        data += _load_images_from_dir(normal_dir, 0, img_size)
        data += _load_images_from_dir(pneumonia_dir, 1, img_size)
    else:
        if s3_client is None or bucket_name is None:
            raise RuntimeError("Dados não encontrados localmente e s3_client/bucket_name não fornecidos.")
        # ajusta os prefixes conforme seu layout no bucket
        data += load_images_from_bucket(s3_client, bucket_name, "Normal/", 0, img_size)
        data += load_images_from_bucket(s3_client, bucket_name, "Pneumonia/", 1, img_size)

    shuffle(data)
    if not data:
        return np.empty((0, img_size, img_size, 3)), np.empty((0,), dtype=int)

    X = np.stack([item[0] for item in data])
    y = np.array([item[1] for item in data], dtype=int)
    return X, y
