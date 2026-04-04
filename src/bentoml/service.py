import typing as t
from PIL.Image import Image
import bentoml


@bentoml.service(
    name="ghisso_classify_fruits",
    traffic={
        "timeout": 300,
    },
)
class Resnet:

    def __init__(self) -> None:
        from pathlib import Path
        import torch
        from clearml import Model
        from timm import create_model
        from timm.data import resolve_model_data_config, create_transform

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = Model.query_models(
                project_name="ImageClassifier",
                model_name="ImageClassifier",
                tags=["best"],
                max_results=1,
            )[0].id
        clearml_model = Model(model_id)
        model_path = Path(clearml_model.get_local_copy())
        metadata = clearml_model.get_all_metadata_casted()
        self.model = create_model(**metadata, checkpoint_path=model_path)
        self.model.to(self.device)
        data_config = resolve_model_data_config(self.model)
        self.inference_transforms = create_transform(**data_config, is_training=False)
        self.label2id = clearml_model.labels
        self.id2label = {v: k for k, v in self.label2id.items()}

        print("Model loaded on device:", self.device)

    @bentoml.api(batchable=True)
    async def classify(self, images: t.List[Image]) -> t.List[dict[str, float]]:
        """
        Classify input images to labels
        """
        import torch

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs)

        labels = []
        for max_possible in logits.argmax(-1):
            label_id = max_possible.item()
            labels.append(self.model.config.id2label[label_id])

        return labels
