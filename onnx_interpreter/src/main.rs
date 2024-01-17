use std::collections::HashMap;
use std::collections::HashSet;

use onnx_interpreter::interpreter::execute_graph;
use onnx_interpreter::onnx::*;
use onnx_interpreter::file;
use onnx_interpreter::utils::*;

fn main() {
    //Read the model
    let path = "models/mobilenet_v3_small_Opset18.onnx";
    //let path = "models/resnet101-v1-7.onnx";
    //let path = "models/vgg16-12.onnx";
    let model = file::read::<ModelProto>(path).unwrap();
    let graph = model.graph.unwrap();

    let inputs = get_inputs(&graph).unwrap();

    // for (name, tensor) in &inputs {
    //     println!("Input: {:?}", name);
    // }

    let ops = graph.node.iter().map(|node| node.op_type.clone()).collect::<HashSet<String>>();
    println!("Operations: {:?}", ops);

    //Print outputs
    //outputs.iter().for_each(|(name, tensor)| println!("Output: {:?} \n{:?}", name, tensor));

}
