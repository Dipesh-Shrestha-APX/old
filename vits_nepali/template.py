# Placeholder file
# template.py
import os

if __name__ == "__main__":
    project_name = "vits-nepali"
    list_of_files = [
        f"{project_name}/configs/config.yaml",
        f"{project_name}/configs/schema.yaml",
        f"{project_name}/data/__init__.py",
        f"{project_name}/data/dataset.py",
        f"{project_name}/data/preprocess.py",
        f"{project_name}/models/__init__.py",
        f"{project_name}/models/text_encoder.py",
        f"{project_name}/models/posterior_encoder.py",
        f"{project_name}/models/flow.py",
        f"{project_name}/models/duration.py",
        f"{project_name}/models/hifigan.py",
        f"{project_name}/models/discriminator.py",
        f"{project_name}/utils/__init__.py",
        f"{project_name}/utils/audio.py",
        f"{project_name}/utils/text.py",
        f"{project_name}/utils/logging.py",
        f"{project_name}/tests/__init__.py",
        f"{project_name}/tests/test_models.py",
        f"{project_name}/pipeline/__init__.py",
        f"{project_name}/pipeline/training_pipeline.py",
        f"{project_name}/pipeline/prediction_pipeline.py",
        f"{project_name}/template.py",
        f"{project_name}/train.py",
        f"{project_name}/inference.py",
        f"{project_name}/requirements.txt",
        f"{project_name}/Dockerfile",
        f"{project_name}/.dockerignore",
        f"{project_name}/setup.py",
        f"{project_name}/README.md",
        f"{project_name}/data/csv/manifest.csv",
        f"{project_name}/data/dataset/audio/.placeholder",
        f"{project_name}/data/dataset/train/.placeholder",
        f"{project_name}/data/dataset/val/.placeholder",
        f"{project_name}/data/dataset/test/.placeholder"
    ]

    for file_path in list_of_files:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write("# Placeholder file\n")
            print(f"Created: {file_path}")
        else:
            print(f"Skipped (already exists): {file_path}")

    print("File structure created successfully!")