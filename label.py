
import json
import random
import shutil
from pathlib import Path

# ================= CONFIGURACIÓN =================

SOURCE_DIR = Path("...") # modificar!!
OUTPUT_DIR = Path("...") # modificar!!

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42

# Clase 0: matrícula
CLASS_ID = 0

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# =================================================


def convert_json_to_yolo(json_path):
    """Convierte las matrículas de un JSON a etiquetas YOLO."""

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_width = data["imageWidth"]
    img_height = data["imageHeight"]

    labels = []

    for lp in data.get("lps", []):

        points = lp.get("poly_coord", [])

        if not points:
            continue

        # Coordenadas del polígono
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        # Bounding box
        x_min = max(0, min(x_coords))
        x_max = min(img_width, max(x_coords))

        y_min = max(0, min(y_coords))
        y_max = min(img_height, max(y_coords))

        if x_max <= x_min or y_max <= y_min:
            continue

        # Centro y dimensiones
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        width = x_max - x_min
        height = y_max - y_min

        # Normalización
        x_center /= img_width
        y_center /= img_height

        width /= img_width
        height /= img_height

        # Formato YOLO:
        # class x_center y_center width height
        label = (
            f"{CLASS_ID} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

        labels.append(label)

    return labels


def create_directories():

    for split in ["train", "val", "test"]:

        (OUTPUT_DIR / split / "images").mkdir(
            parents=True, exist_ok=True
        )

        (OUTPUT_DIR / split / "labels").mkdir(
            parents=True, exist_ok=True
        )


def main():

    random.seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    create_directories()

    # Buscar imágenes
    image_files = [
        f for f in SOURCE_DIR.iterdir()
        if f.is_file()
        and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # Buscar parejas imagen-JSON
    dataset_pairs = []
    missing_json = []

    for image_path in image_files:

        json_path = image_path.with_suffix(".json")

        if json_path.exists():
            dataset_pairs.append((image_path, json_path))
        else:
            missing_json.append(image_path.name)

    print(f"Imágenes encontradas: {len(image_files)}")
    print(f"Parejas válidas: {len(dataset_pairs)}")
    print(f"Sin JSON: {len(missing_json)}")

    if not dataset_pairs:
        print("No se encontraron parejas imagen-JSON.")
        return

    # Barajar de forma reproducible
    random.shuffle(dataset_pairs)

    total = len(dataset_pairs)

    # División del dataset
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": dataset_pairs[:train_end],
        "val": dataset_pairs[train_end:val_end],
        "test": dataset_pairs[val_end:]
    }

    # Procesar cada división
    for split_name, pairs in splits.items():

        print(f"\nProcesando {split_name}: {len(pairs)} imágenes")

        for image_path, json_path in pairs:

            # Destinos
            image_output = (
                OUTPUT_DIR / split_name / "images" / image_path.name
            )

            label_output = (
                OUTPUT_DIR / split_name / "labels"
                / f"{image_path.stem}.txt"
            )

            # Copiar imagen
            shutil.copy2(image_path, image_output)

            # Convertir anotaciones
            labels = convert_json_to_yolo(json_path)

            # Guardar etiquetas
            with open(label_output, "w", encoding="utf-8") as f:
                f.write("\n".join(labels))

    # Crear data.yaml
    yaml_content = f"""\
        path: {OUTPUT_DIR.resolve().as_posix()}

        train: images/train
        val: images/val
        test: images/test

        names:
        0: license_plate
    """

    yaml_path = OUTPUT_DIR / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("\n==============================")
    print("Dataset creado correctamente")
    print(f"Train: {len(splits['train'])}")
    print(f"Val:   {len(splits['val'])}")
    print(f"Test:  {len(splits['test'])}")
    print(f"Salida: {OUTPUT_DIR.resolve()}")
    print(f"YAML:   {yaml_path.resolve()}")


if __name__ == "__main__":
    main()
