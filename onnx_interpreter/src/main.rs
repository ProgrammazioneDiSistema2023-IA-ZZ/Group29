use std::fs::File;
use std::io::Write;
use std::time::Instant;

use onnx_interpreter::interpreter::execute_graph;
use onnx_interpreter::onnx::*;
use onnx_interpreter::file;
use onnx_interpreter::input_for_mnist::input_for_mnist;
use onnx_interpreter::utils::*;

fn main() {
    //Read the model
    // let model_name = "shufflenet-v2-12";
    let model_name = "mnist-12";

    let model = file::read::<ModelProto>(&format!("models/{}.onnx", model_name)).unwrap();
    let graph = model.graph.unwrap(); 

    let user_input = match model_name {
        "mnist-12" => Some(input_for_mnist()),
        _ => None
    };

    let mut log_file = File::create(&format!("logs/{}.txt", model_name)).unwrap();
    log_file.write(format!("Model: {}\n", model_name).as_bytes()).unwrap();

    let mut inputs = get_inputs(&graph, user_input).unwrap();

    for input in graph.input.iter() {
        let (name, array) = inputs.get_key_value(&input.name).unwrap();
        println!("Input: {:?} with shape {:?}:\n{:?}\n\n", name, array.shape(), array);
    }

    //Write the inputs info
    for input in graph.input.iter() {
        let (name, array) = inputs.get_key_value(&input.name).unwrap();
        log_file.write(format!("Input: {:?} with shape {:?}:\n{:?}\n\n", name, array.shape(), array).as_bytes()).unwrap();
    }

    let start = Instant::now();
    let (outputs, info) = execute_graph(&graph, &mut inputs, true).unwrap();
    let duration = start.elapsed();

    //Write the outputs and nodes info
    for (node_name, node_type, node_input, node_input_shape, node_output, node_output_shape, duration) in info {
        log_file.write(format!("Node: {:?} execute {:?} in {:?}:\n\tInput: ", node_name, node_type, duration).as_bytes()).unwrap();
        for (name, array) in node_input.iter().zip(node_input_shape.iter()) {
            log_file.write(format!("{:?} with shape {:?} - ", name, array).as_bytes()).unwrap();
        }
        log_file.write(format!("\n\tOutput: ").as_bytes()).unwrap();
        for (name, array) in node_output.iter().zip(node_output_shape.iter()) {
            log_file.write(format!("{:?} with shape {:?} - ", name, array).as_bytes()).unwrap();
        }
        log_file.write(format!("\n\n").as_bytes()).unwrap();
    }

    for (name, array) in outputs.iter() {
        log_file.write(format!("Output: {:?} with shape {:?}:\n{:?}\n\n", name, array.shape(), array).as_bytes()).unwrap();
        if model_name == "mnist-12" {
            let index = array.arg_max();
            log_file.write(format!("Predicted number: {:?}\n\n", index).as_bytes()).unwrap();
        }
    }

    // Print outputs
    println!("Execution time: {:?}", duration);

    match model_name {
        "mnist-12" => {
            outputs.iter().for_each(|(name, tensor)| {
                println!("Output: {:?} \n{:?}", name, tensor);
                let index = tensor.arg_max();
                println!("Predicted number: {:?}", index);
            })
        },
        _ =>  outputs.iter().for_each(|(name, tensor)| println!("Output: {:?} \n{:?}", name, tensor))
    }
 
}

