use std::collections::HashMap;
use ndarray::*;

use crate::onnx::*;
use crate::operations::*;  


pub fn execute_node(node: &NodeProto, inputs:  &HashMap<String, Array2<f32>>) -> HashMap<String, Array2<f32>> {
    let mut outputs = HashMap::new();
    let op_type = &node.op_type;
    let input_tensors = node.input.iter().map(|input| inputs.get(input).unwrap()).collect::<Vec<&Array2<f32>>>();
    match op_type.as_str() {
        "Add" => {
            let output = add_tensors(input_tensors[0], input_tensors[1]);
            outputs.insert(node.output[0].clone(), output);
        },
        "MatMul" => {
            let output = matmul_tensors(input_tensors[0], input_tensors[1]);
            outputs.insert(node.output[0].clone(), output);
        }
        _ => panic!("Operation not supported")            
    }

    // Print node information
    println!("Node: {:?}", node.op_type);
    input_tensors.iter().for_each(|tensor| println!("Input: \n{:?}", tensor));
    outputs.iter().for_each(|(name, tensor)| println!("Output: {:?} \n{:?}", name, tensor));
    outputs
}

pub fn execute_graph(graph: &GraphProto, inputs: &mut HashMap<String, Array2<f32>>) -> HashMap<String, Array2<f32>> {
    for node in graph.node.iter() {
        let node_outputs = execute_node(&node, inputs);
        inputs.extend(node_outputs);
    }
    let outputs = graph.output.iter().map(|output| (output.name.clone(), inputs.get(&output.name).unwrap().clone())).collect::<HashMap<String, Array2<f32>>>();

    outputs
}
