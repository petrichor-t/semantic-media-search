from sentence_transformers import SentenceTransformer


class ClipModel:

    MODEL_NAME = "sentence-transformers/clip-ViT-B-32"

    def __init__(self) -> None:
        self.model = SentenceTransformer(
            self.MODEL_NAME
        )