use std::collections::HashMap;
use std::collections::HashSet;

use onnx_interpreter::interpreter::execute_graph;
use onnx_interpreter::onnx::*;
use onnx_interpreter::file;
use onnx_interpreter::utils::*;

fn main() {
    //Read the model
    let path = "models/mobilenetv2-12.onnx";
    // let path = "models/shufflenet-v2-12.onnx";

    let model = file::read::<ModelProto>(path).unwrap();
    let graph = model.graph.unwrap();

    let inputs = get_inputs(&graph).unwrap();

    // for (name, tensor) in &inputs {
    //     println!("Input: {:?}", name);
    // }

    let mut conv = true; 
    let mut concat = true;
    for node in graph.node.iter() {
        if node.op_type == "Conv" && conv {
            println!("Conv: {:?}", node);
            conv = false;
        } else if node.op_type == "Concat" && concat {
            println!("Concat: {:?}", node);
            concat = false;
        }
    }

    //Print outputs
    //outputs.iter().for_each(|(name, tensor)| println!("Output: {:?} \n{:?}", name, tensor));

}
