import torch
import onnx
import numpy as np
import argparse
import onnx_graphsurgeon as gs
from post_processing import *

def post_process_packnet(model_file, opset=11):
    """
    Use ONNX graph surgeon to replace upsample and instance normalization nodes. Refer to post_processing.py for details.
    Args:
        model_file : Path to ONNX file
    """
    # Load the packnet graph
    graph = gs.import_onnx(onnx.load(model_file))

    if opset==11:
        graph = process_pad_nodes(graph)

    # Replace the subgraph of upsample with a single node with input and scale factor.
    #graph = process_upsample_nodes(graph, opset)

    # Convert the group normalization subgraph into a single plugin node.
    graph = process_groupnorm_nodes(graph)

    # Remove unused nodes, and topologically sort the graph.
    graph.cleanup().toposort()

    # Export the onnx graph from graphsurgeon
    onnx.save_model(gs.export_onnx(graph), '/models/run08/jetracerpost.onnx')

    print("Saving the ONNX model to {}".format(model_file))

def main():
    parser = argparse.ArgumentParser(description="post-processes onnx model it to insert TensorRT plugins")
    parser.add_argument("-o", "--output", help="Path to save the generated ONNX model", default="/models/run08/jetracer.onnx")
    parser.add_argument("-op", "--opset", type=int, help="ONNX opset to use", default=11)
    parser.add_argument("-v", "--verbose", action='store_true', help="Flag to enable verbose logging for torch.onnx.export")
    args=parser.parse_args()

    # Perform post processing on Instance Normalization and upsampling nodes and create a new ONNX graph
    post_process_packnet(args.output, args.opset)

if __name__ == '__main__':
    main()


