use std::collections::HashMap;
use std::thread;
use std::sync::{Arc, Mutex, Condvar};

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
            println!("Conv {:?}", attributes.keys());
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
            let output = ArrayMultiType::convolution(input_tensors[0], input_tensors[1], bias, Some(auto_pad.as_str()), &dilations, group, &kernel_shape, Some(pads.as_slice()), &strides);
            outputs.insert(node.output[0].clone(), output)
        },
        "Relu" => outputs.insert(node.output[0].clone(), ArrayMultiType::relu(input_tensors[0])),
        "Transpose" => {
            let perm = match attributes.get("perm") {
                Some(Attribute::Ints(perm)) => Some(perm.iter().map(|x| *x as usize).collect::<Vec<usize>>()),
                _ => None
            };
            outputs.insert(node.output[0].clone(), ArrayMultiType::transpose(input_tensors[0], perm))
        },
        "Tile" => {
            let repeats = match attributes.get("repeats") {
                Some(Attribute::Ints(repeats)) => repeats.iter().map(|x| *x as usize).collect::<Vec<usize>>(),
                _ => return Err("Invalid repeats")
            };
            outputs.insert(node.output[0].clone(), ArrayMultiType::tile(input_tensors[0], &repeats))
        },
        "Gather" => {
            let axis = match attributes.get("axis") {
                Some(Attribute::Int(axis)) => *axis as usize,
                _ => return Err("Invalid axis")
            };
            let indices = &input_tensors[1].to_vec_usize();
            outputs.insert(node.output[0].clone(), ArrayMultiType::gather(input_tensors[0], indices, axis))
        },
        "Constant" => {
            let value = match attributes.get("value") {
                Some(Attribute::Tensor(value)) => value,
                _ => return Err("Invalid value")
            };
            outputs.insert(node.output[0].clone(), value.clone())
        },
        "Reshape" => {
            let shape = input_tensors[1].to_vec_i64();
            outputs.insert(node.output[0].clone(), ArrayMultiType::reshape(input_tensors[0], &shape))
        },
        "Gemm" => {
            let alpha = match attributes.get("alpha") {
                Some(Attribute::Float(alpha)) => *alpha,
                _ => return Err("Invalid alpha")
            };
            let beta = match attributes.get("beta") {
                Some(Attribute::Float(beta)) => *beta,
                _ => return Err("Invalid beta")
            };
            let trans_a = match attributes.get("transA") {
                Some(Attribute::Int(a)) => *a != 0,
                _ => return Err("Invalid transA")
            };
            let trans_b = match attributes.get("transB") {
                Some(Attribute::Int(b)) => *b != 0,
                _ => return Err("Invalid transB")
            };
            let bias = match input_tensors.len() {
                3.. => Some(input_tensors[2]),
                _=> None
            };

            outputs.insert(node.output[0].clone(), ArrayMultiType::gemm(input_tensors[0], input_tensors[1], bias, alpha, beta, trans_a, trans_b))
        },
        "Clip" => {
            let min = match input_tensors.len() {
                2.. => Some(input_tensors[1]),
                _=> None
            };
            let max = match input_tensors.len() {
                3.. => Some(input_tensors[2]),
                _=> None
            };

            outputs.insert(node.output[0].clone(), ArrayMultiType::clip(input_tensors[0], min, max))
        },
        "Unsqueeze" => {
            let axes = input_tensors[1].to_vec_i64();
            outputs.insert(node.output[0].clone(), ArrayMultiType::unsqueeze(input_tensors[0], &axes))
        },
        _ => return Err("Operation not supported")        
    };
    // Print node information
    // println!("Node: {:?}", node.op_type);
    // input_tensors.iter().for_each(|tensor| println!("Input: \n{:?}", tensor));
    // outputs.iter().for_each(|(name, tensor)| println!("Output: {:?} \n{:?}", name, tensor));

    Ok(outputs)
}

pub fn execute_graph(graph: &GraphProto, inputs: &mut HashMap<String, ArrayMultiType>, verbose: bool) -> Result<HashMap<String, ArrayMultiType>, &'static str> {
    let shared_inputs = Arc::new((Mutex::new(inputs.clone()), Condvar::new()));
    let mut threads = Vec::new();
    
    for node in graph.node.iter() {
        let node_clone = node.clone();
        let verbose_clone = verbose.clone();
        let shared_inputs_clone = Arc::clone(&shared_inputs);
        threads.push(thread::spawn(move || {
            if verbose {
                println!("{:?} - {:?}: Start node", node_clone.name, node_clone.op_type);
            }
            // Aquiring the lock
            let (lock, cvar) = &*shared_inputs_clone;
            let mut inputs = lock.lock().unwrap();

            if verbose_clone {
                //println!("{:?} - {:?}: Wait inputs: {:?}", node_clone.name, node_clone.op_type, node_clone.input);
            }
            //Check if all inputs are available, if not relese the lock and wait for the condition variable
            while node_clone.input.iter().any(|input| !inputs.contains_key(input)) {
                inputs = cvar.wait(inputs).unwrap();
            }
            // Get the inputs and relese the lock
            let inputs_clone = inputs.clone();
            drop(inputs);

            if verbose_clone {
                println!("{:?} - {:?}: Execute node", node_clone.name, node_clone.op_type);
            }
            // Execute the nod
            let node_outputs = execute_node(&node_clone, &inputs_clone).unwrap();
            
            // Aquiring the lock
            let (lock, cvar) = &*shared_inputs_clone;
            let mut inputs = lock.lock().unwrap();

            // Update the inputs and notify the condition variable
            if verbose {
                println!("{:?} - {:?}: Update inputs: {:?}", node_clone.name, node_clone.op_type, node_outputs.keys());
            }
            inputs.extend(node_outputs);
            cvar.notify_all();
        }));
    }

    for thread in threads {
        thread.join().unwrap();
    }

    let (lock, _) = &*shared_inputs;
    let inputs = lock.lock().unwrap();
    let outputs = graph.output.iter().map(|output| (output.name.clone(), inputs.get(&output.name).unwrap().clone())).collect::<HashMap<String, ArrayMultiType>>();

    Ok(outputs)
}
