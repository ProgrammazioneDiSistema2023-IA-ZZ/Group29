use std::collections::HashSet;

use onnx_interpreter::interpreter::execute_graph;
use onnx_interpreter::onnx::*;
use onnx_interpreter::file;
use onnx_interpreter::utils::*;

fn main() {
    //Read the model
    let path = "models/mobilenetv2-12.onnx";
    // let path = "models/shufflenet-v2-12.onnx";
    // let path = "models/linear_regression.onnx";

    let model = file::read::<ModelProto>(path).unwrap();
    let graph = model.graph.unwrap();

    let mut inputs = get_inputs(&graph).unwrap();

    // for (name, input) in inputs.iter() {
    //     println!("Input: {:?}, shape: {:?}", name, input.shape());
    // }

    let distinc_op = graph.node.iter().map(|node| node.op_type.clone()).collect::<HashSet<String>>();
    println!("Distinct op: {:?}", distinc_op);

    let outputs = execute_graph(&graph, &mut inputs, true).unwrap();

    // Print outputs
    //outputs.iter().for_each(|(name, tensor)| println!("Output: {:?} \n{:?}", name, tensor));

}
