from unittest.mock import MagicMock

from taxonomic_rag_system.utils.vision_models import Qwen3VLDescriptiveCaptionerLocal


def test_qwen3vl_captioner_generate_caption_smoke() -> None:
    captioner = Qwen3VLDescriptiveCaptionerLocal(model_id="Qwen/Qwen3-VL-8B-Instruct")

    mock_model = MagicMock()
    # input_ids = [[1,2]]; output ids include prompt then generated tokens
    mock_model.generate.return_value = [[1, 2, 42, 43]]

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = {
        "input_ids": [[1, 2]],
        "token_type_ids": [[0, 0]],
    }
    mock_processor.batch_decode.return_value = ["mock caption output"]

    def _fake_lazy_load() -> None:
        captioner._model = mock_model
        captioner._processor = mock_processor

    captioner._lazy_load = _fake_lazy_load  # type: ignore[method-assign]

    # Image is not inspected by the mocked processor; pass any sentinel.
    out = captioner.generate_caption(image=MagicMock())
    assert out == "mock caption output"
    assert mock_processor.apply_chat_template.call_count == 1
    assert mock_model.generate.call_count == 1

