use std::collections::HashMap;

use crate::onnx::*;
use crate::array::ArrayMultiType;

pub fn execute_node(node: &NodeProto, inputs:  &HashMap<String, ArrayMultiType>) -> Result<HashMap<String, ArrayMultiType>, &'static str> {
    let mut outputs = HashMap::new();
    let op_type = &node.op_type;
    let input_tensors = node.input.iter().map(|input| inputs.get(input).unwrap()).collect::<Vec<&ArrayMultiType>>();
    match op_type.as_str() {
        "Mul" => outputs.insert(node.output[0].clone(), ArrayMultiType::multiply(input_tensors[0], input_tensors[1])),
        "Add" => outputs.insert(node.output[0].clone(), ArrayMultiType::add(input_tensors[0], input_tensors[1])),
        "Greater" => outputs.insert(node.output[0].clone(), ArrayMultiType::greater(input_tensors[0], input_tensors[1])),
        "GreaterOrEqual" => outputs.insert(node.output[0].clone(), ArrayMultiType::greater_or_equal(input_tensors[0], input_tensors[1])),
        "Less" => outputs.insert(node.output[0].clone(), ArrayMultiType::less(input_tensors[0], input_tensors[1])),
        "LessOrEqual" => outputs.insert(node.output[0].clone(), ArrayMultiType::less_or_equal(input_tensors[0], input_tensors[1])),
        "MatMul" => outputs.insert(node.output[0].clone(), ArrayMultiType::matmul(input_tensors[0], input_tensors[1])),
        _ => return Err("Operation not supported")        
    };
    // Print node information
    // println!("Node: {:?}", node.op_type);
    // input_tensors.iter().for_each(|tensor| println!("Input: \n{:?}", tensor));
    // outputs.iter().for_each(|(name, tensor)| println!("Output: {:?} \n{:?}", name, tensor));

    Ok(outputs)
}

pub fn execute_graph<A, D>(graph: &GraphProto, inputs: &mut HashMap<String, ArrayMultiType>) -> Result<HashMap<String, ArrayMultiType>, &'static str> {
    let mut outputs = HashMap::new();
    
    for node in graph.node.iter() {
        let node_outputs = execute_node(&node, inputs)?;

        for (name, output) in node_outputs {
            if graph.output.iter().any(|output| output.name == name) {
                outputs.insert(name, output);
            } else {
                inputs.insert(name, output);
            }
        }
    }

    Ok(outputs)
}
