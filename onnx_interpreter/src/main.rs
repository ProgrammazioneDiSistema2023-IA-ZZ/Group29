use std::collections::HashMap;
use ndarray::Array2;

use onnx_interpreter::interpreter::execute_graph;
use onnx_interpreter::onnx::*;
use onnx_interpreter::file;

fn main() {
    //Example of write and read any object
    let path = "models/test.onnx";
    let mut x = ModelProto::default();
    x.producer_name = "test".to_string();

    //Write
    println!("Write object: \n{:?}\n", x);
    file::write(&x, path).unwrap();

    //Read
    let y = file::read::<ModelProto>(path).unwrap();
    println!("Read object: \n{:?}\n", y);    

    //Example of read linear regression file
    let path = "models/linear_regression.onnx";
    let model = file::read::<ModelProto>(path).unwrap();

    println!("Linear regression model: \n{:?}\n", model.graph);

    //Example of execute graph
    let mut inputs = HashMap::new();
    inputs.insert("X".to_string(), Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    inputs.insert("A".to_string(), Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    inputs.insert("B".to_string(), Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap());

    let outputs = execute_graph(&model.graph.unwrap(), &mut inputs);

    //Print outputs
    outputs.iter().for_each(|(name, tensor)| println!("Output: {:?} \n{:?}", name, tensor));

}
