use std::collections::HashMap;
use ndarray::*;

use crate::onnx::*;
use crate::operations::*;  


pub fn execute_node(node: &NodeProto, inputs: &HashMap<String, Array2<f32>>) -> HashMap<String, Array2<f32>> {
    let mut outputs = HashMap::new();

    match node.op_type.as_str() {
        "Add" => {
            let a = inputs.get(&node.input[0]).unwrap();
            let b = inputs.get(&node.input[1]).unwrap();
            let result = add_tensors(a, b);  // Assicurati che `add` sia definita correttamente in `operations`
            outputs.insert(node.output[0].clone(), result.clone());
        },
        "Relu" => {
            let a = inputs.get(&node.input[0]).unwrap();
            let result = relu(a); 
            outputs.insert(node.output[0].clone(), result.clone());
        },
        "Concat" => {
            // Gestione dell'operatore Concat
            // ...
        },
        // Altri operatori...
        _ => panic!("Operatore non supportato: {}", node.op_type),
    }

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
