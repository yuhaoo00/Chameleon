
import os
import onnx
import tempfile
from copy import deepcopy
from pathlib import Path
import numpy as np
import onnx_graphsurgeon as gs
from polygraphy.backend.onnx.loader import fold_constants

def add_groupnorm(graph):
    cnt = 0
    for node in graph.nodes:
        if node.op == "Reshape" and node.o().op == "InstanceNormalization" and node.o().o().op == "Reshape" \
                and node.o().o().o().op == "Mul" and node.o().o().o().o().op == "Add":

            last_node = node.o().o().o().o()

            instance_norm = node.o()
            instance_norm_scale = instance_norm.inputs[1]
            instance_norm_bias = instance_norm.inputs[2]
            epsilon = instance_norm.attrs["epsilon"]
            mul_node = node.o().o().o()
            add_node = node.o().o().o().o()

            scale = np.ascontiguousarray(np.array(deepcopy(instance_norm_scale.values.tolist()), dtype=np.float16))
            bias = np.ascontiguousarray(np.array(deepcopy(instance_norm_bias.values.tolist()), dtype=np.float16))
            gamma = np.ascontiguousarray(np.array(deepcopy(mul_node.inputs[1].values.tolist()), dtype=np.float16))
            beta = np.ascontiguousarray(np.array(deepcopy(add_node.inputs[1].values.tolist()), dtype=np.float16))

            with_swish = True if node.o().o().o().o().o().o().op == "Sigmoid" and node.o().o().o().o().o().o().o().op == "Mul" else False
            if with_swish:
                last_node = node.o().o().o().o().o().o().o()

            constant_gamma = gs.Constant("gamma_{}".format(cnt), gamma.reshape(-1))
            constant_beta = gs.Constant("beta_{}".format(cnt), beta.reshape(-1))
            x = node.inputs[0]
            group_norm_v = gs.Variable("group_norm_{}".format(cnt), np.dtype(np.float16), x.shape)
            group_norm = gs.Node("GroupNorm", "GroupNorm_{}".format(cnt),
                                attrs={"epsilon": epsilon, "bSwish": with_swish},
                                inputs=[x, constant_gamma, constant_beta],
                                outputs=[group_norm_v])
            cnt += 1
            for n in graph.nodes:
                if last_node.outputs[0] in n.inputs:
                    index = n.inputs.index(last_node.outputs[0])
                    n.inputs[index] = group_norm.outputs[0]
            last_node.outputs = []
            graph.nodes.append(group_norm)

    print("add groupnorm: ", cnt)


def has_external_data(onnx_model_path):
    original_model = onnx.load_model(str(onnx_model_path), load_external_data=False)
    for initializer in original_model.graph.initializer:
        if initializer.HasField("data_location") and initializer.data_location == onnx.TensorProto.EXTERNAL:
            return True
    return False
        

def optimize_onnx(onnx_dir):
    onnx_dir = Path(onnx_dir)

    for onnx_path in onnx_dir.glob("**/*.onnx"):
        onnx_path = onnx_path.as_posix()
        print("[Optimization] target onnx:", onnx_path)
        save_as_external_data = has_external_data(onnx_path)

        # fold_constants
        onnx_graph = fold_constants(onnx.load(onnx_path), allow_onnxruntime_shape_inference=True)
        graph = gs.import_onnx(onnx_graph)
        graph.cleanup().toposort()

        # add_groupnorm
        if "vae" not in onnx_path:
            add_groupnorm(graph)
        graph.cleanup().toposort()

        # shape_inference
        onnx_graph = gs.export_onnx(graph)
        if onnx_graph.ByteSize() > onnx.checker.MAXIMUM_PROTOBUF:
            temp_dir = tempfile.TemporaryDirectory().name
            os.makedirs(temp_dir, exist_ok=True)
            onnx_orig_path = os.path.join(temp_dir, 'model.onnx')
            onnx_inferred_path = os.path.join(temp_dir, 'inferred.onnx')
            onnx.save_model(onnx_graph,
                onnx_orig_path,
                save_as_external_data=save_as_external_data,
                all_tensors_to_one_file=True,
                convert_attribute=False)
            onnx.shape_inference.infer_shapes_path(onnx_orig_path, onnx_inferred_path)
            onnx_graph = onnx.load(onnx_inferred_path)
        else:
            onnx_graph = onnx.shape_inference.infer_shapes(onnx_graph)


        save_path = onnx_path.replace("model.onnx", "model.opt.onnx")

        onnx.save_model(gs.export_onnx(graph),
                        save_path,
                        save_as_external_data=save_as_external_data,
                        all_tensors_to_one_file=True,
                        location="model.opt.onnx.data",
                        convert_attribute=False)
    
