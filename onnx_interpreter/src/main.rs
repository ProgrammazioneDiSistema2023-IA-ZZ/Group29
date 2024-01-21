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

    for (name, input) in inputs.iter() {
        println!("Input: {:?}, shape: {:?}", name, input.shape());
    }

    for node in graph.node.iter() {
        if node.op_type == "Reshape" || node.op_type == "Clip" {
            //println!("Node: {:?}", node);
        }
    }

    let outputs = execute_graph(&graph, &mut inputs, false).unwrap();

    // Print outputs
    //outputs.iter().for_each(|(name, tensor)| println!("Output: {:?} \n{:?}", name, tensor));

}
