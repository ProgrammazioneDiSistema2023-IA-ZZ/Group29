use std::collections::HashSet;
use std::time::{Instant, Duration};

use ndarray::Array;
use onnx_interpreter::interpreter::execute_graph;
use onnx_interpreter::onnx::*;
use onnx_interpreter::file;
use onnx_interpreter::utils::*;

fn main() {
    //Read the model
    // let path = "models/mobilenetv2-12.onnx";
    // let path = "models/shufflenet-v2-12.onnx";
    // let path = "models/linear_regression.onnx";
    // let path = "models/super-resolution-10.onnx";
    // let path = "models/efficientnet-lite4-11.onnx";
    let path = "models/mnist-12.onnx";

    let model = file::read::<ModelProto>(path).unwrap();
    let graph = model.graph.unwrap(); 

    let mut inputs = get_inputs(&graph).unwrap();

    let start = Instant::now();
    let outputs = execute_graph(&graph, &mut inputs, true).unwrap();
    let duration = start.elapsed();

    // Print outputs
    println!("Execution time: {:?}", duration);
    outputs.iter().for_each(|(name, tensor)| println!("Output: {:?} \n{:?}", name, tensor));

}

