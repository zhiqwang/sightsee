import onnx
import pytest
from onnx import TensorProto, helper


@pytest.fixture
def write_test_model():
    def _write_test_model(model_path):
        node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
        graph = helper.make_graph(
            [node],
            "sightsee-test-model",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(graph, producer_name="sightsee-tests")
        onnx.save_model(model, model_path)
        return str(model_path)

    return _write_test_model
