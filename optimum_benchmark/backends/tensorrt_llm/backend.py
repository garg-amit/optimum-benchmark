from collections import OrderedDict
from tempfile import TemporaryDirectory
from typing import Any, Dict

from hydra.utils import get_class

from ..base import Backend
from .config import TRTLLMConfig
from .utils import MODEL_TYPE_TO_TRTLLMMODEL
from tensorrt_llm import SamplingParams


class TRTLLMBackend(Backend[TRTLLMConfig]):
    NAME = "tensorrt-llm"

    def __init__(self, config: TRTLLMConfig):
        super().__init__(config)

        if self.config.model_type in MODEL_TYPE_TO_TRTLLMMODEL:
            self.trtllm_loader = get_class(MODEL_TYPE_TO_TRTLLMMODEL[self.config.model_type])
            self.logger.info(f"\t+ Using TRTLLMModel class {self.trtllm_loader.__name__}")
        else:
            raise NotImplementedError(f"TRTLLMBackend does not support model_type {self.config.model_type}")

    def load(self) -> None:
        self.logger.info("\t+ Creating backend temporary directory")
        self.tmpdir = TemporaryDirectory()

        self.logger.info("\t+ Loading pretrained TRTLLMModel")
        self.load_trtmodel_from_pretrained()

        self.logger.info("\t+ Cleaning up backend temporary directory")
        self.tmpdir.cleanup()

    def load_trtmodel_from_pretrained(self) -> None:
        self.pretrained_model = self.trtllm_loader(
            model=self.config.model
        )

    def prefill(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        params = SamplingParams(
            min_tokens=kwargs.get("min_new_tokens", -1),
            max_tokens=kwargs.get("max_new_tokens", -1),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", 50),
            seed=kwargs.get("seed", 42),
        )
        return self.pretrained_model.generate(inputs.get("input_ids").tolist(), params)

    def generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        params = SamplingParams(
            min_tokens=kwargs.get("min_new_tokens", -1),
            max_tokens=kwargs.get("max_new_tokens", -1),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", 50),
            seed=kwargs.get("seed", 42),
        )
        return self.pretrained_model.generate(inputs.get("input_ids").tolist(), params)
