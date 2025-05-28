"""Secondary target: TFLite via torch → ONNX → TF → TFLite.

Demonstrates the path to the broader Android / embedded Linux ecosystem.
Note: needs `tensorflow` and `onnx-tf` in the deploy extra.
"""
from __future__ import annotations
from pathlib import Path
import tempfile


def export(onnx_path: str, out_path: str, calibration_data=None):
    from onnx_tf.backend import prepare
    import onnx
    import tensorflow as tf

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    with tempfile.TemporaryDirectory() as td:
        sm = Path(td) / "saved_model"
        tf_rep.export_graph(str(sm))

        converter = tf.lite.TFLiteConverter.from_saved_model(str(sm))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if calibration_data is not None:
            def rep_dataset():
                for batch in calibration_data:
                    yield [batch]
            converter.representative_dataset = rep_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        tflite_bytes = converter.convert()

    Path(out_path).write_bytes(tflite_bytes)
    return out_path
