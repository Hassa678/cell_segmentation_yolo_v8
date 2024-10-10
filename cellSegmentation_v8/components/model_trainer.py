import subprocess
import zipfile
import os
import sys
from pathlib import Path
from cellSegmentation_v8.utils.main_utils import read_yaml_file
from cellSegmentation_v8.logger import logging
from cellSegmentation_v8.exception import AppException
from cellSegmentation_v8.entity.config_entity import ModelTrainerConfig
from cellSegmentation_v8.entity.artifacts_entity import ModelTrainerArtifact
import shutil

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            # Unzipping data
            logging.info("Unzipping data.zip")
            with zipfile.ZipFile("data.zip", "r") as zip_ref:
                zip_ref.extractall()  # Extracts in the current directory
            os.remove("data.zip")  # Removes the zip file

            # Running YOLO training
            yolo_cmd = [
                "yolo",
                "task=segment",
                "mode=train",
                f"model={self.model_trainer_config.weight_name}",
                "data=data.yaml",
                f"epochs={self.model_trainer_config.no_epochs}",
                "imgsz=640",
                "save=true"
            ]
            subprocess.run(yolo_cmd, check=True)

            # Creating directory for trained model if it doesn't exist
            destination_dir = Path(self.model_trainer_config.model_trainer_dir)
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination_path = destination_dir / "best.pt"
            source_path = Path(r"C:\cell_segmentation_yolo_v8\runs\segment\train\weights\best.pt")

            # Move the best model file to the designated directory
            shutil.copy(str(source_path), str(destination_path))

            # Cleaning up unnecessary files
            files_to_remove = [
                "yolov8s-seg.pt", "train", "valid", "test", "data.yaml", "runs"
            ]
            for file_or_dir in files_to_remove:
                path = Path(file_or_dir)
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()

            # Returning the artifact with the trained model file path
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=str(destination_path),
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise AppException(e, sys)