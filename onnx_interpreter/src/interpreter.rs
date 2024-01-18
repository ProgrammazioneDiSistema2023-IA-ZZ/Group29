use std::collections::HashMap;

use crate::onnx::*;
use crate::utils::*;
use crate::attribute::Attribute;
use crate::array::ArrayMultiType;

pub fn execute_node(node: &NodeProto, inputs:  &HashMap<String, ArrayMultiType>) -> Result<HashMap<String, ArrayMultiType>, &'static str> {
    let mut outputs = HashMap::new();
    let op_type = &node.op_type;
    let input_tensors = node.input.iter().map(|input| inputs.get(input).unwrap()).collect::<Vec<&ArrayMultiType>>();
    let attributes = get_attributes(node)?;
    match op_type.as_str() {
        "Mul" => outputs.insert(node.output[0].clone(), ArrayMultiType::multiply(input_tensors[0], input_tensors[1])),
        "Add" => outputs.insert(node.output[0].clone(), ArrayMultiType::add(input_tensors[0], input_tensors[1])),
        "Greater" => outputs.insert(node.output[0].clone(), ArrayMultiType::greater(input_tensors[0], input_tensors[1])),
        "GreaterOrEqual" => outputs.insert(node.output[0].clone(), ArrayMultiType::greater_or_equal(input_tensors[0], input_tensors[1])),
        "Less" => outputs.insert(node.output[0].clone(), ArrayMultiType::less(input_tensors[0], input_tensors[1])),
        "LessOrEqual" => outputs.insert(node.output[0].clone(), ArrayMultiType::less_or_equal(input_tensors[0], input_tensors[1])),
        "MatMul" => outputs.insert(node.output[0].clone(), ArrayMultiType::matmul(input_tensors[0], input_tensors[1])),
        "GlobalAveragePool" => outputs.insert(node.output[0].clone(), ArrayMultiType::global_average_pool(input_tensors[0])),
        "Concat" => {
            let axis = match attributes.get("axis") {
                Some(Attribute::Int(axis)) => *axis as isize,
                _ => return Err("Invalid axis")
            };
            outputs.insert(node.output[0].clone(), ArrayMultiType::concat(input_tensors, axis))
        },
        "Conv" => {
            let kernel_shape = match attributes.get("kernel_shape") {
                Some(Attribute::Ints(kernel_shape)) => kernel_shape,
                _ => return Err("Invalid kernel shape")
            };
            let strides = match attributes.get("strides") {
                Some(Attribute::Ints(strides)) => strides,
                _ => return Err("Invalid strides")
            };
            let dilations = match attributes.get("dilations") {
                Some(Attribute::Ints(dilations)) => dilations,
                _ => return Err("Invalid dilations")
            };
            let pads = match attributes.get("pads") {
                Some(Attribute::Ints(pads)) => pads,
                _ => return Err("Invalid pads")
            };
            let group = match attributes.get("group") {
                Some(Attribute::Int(group)) => *group,
                _ => return Err("Invalid group")
            };
            let default_auto_pad = "NOTSET".to_string();
            let auto_pad = match attributes.get("auto_pad") {
                Some(Attribute::String(auto_pad)) => auto_pad,
                _ => &default_auto_pad
            };
            let bias = match input_tensors.len() {
                3.. => Some(input_tensors[2]),
                _=> None
            };
            let output = ArrayMultiType::convolution(input_tensors[0], input_tensors[1], bias, auto_pad, dilations, group, kernel_shape, pads, strides);
            outputs.insert(node.output[0].clone(), output)
        },
        "Relu" => outputs.insert(node.output[0].clone(), ArrayMultiType::relu(input_tensors[0])),

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
