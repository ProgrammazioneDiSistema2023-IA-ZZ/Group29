use std::time::{Instant, Duration};
use std::collections::HashMap;
use std::{thread, default};
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
            let default_kernel_shape = input_tensors[1].shape()[2..].iter().map(|&x| x as i64).collect::<Vec<i64>>();
            let kernel_shape = match attributes.get("kernel_shape") {
                Some(Attribute::Ints(kernel_shape)) => kernel_shape,
                _ => &default_kernel_shape
            };
            let default_strides = input_tensors[0].shape()[2..].iter().map(|_| 1).collect::<Vec<i64>>();
            let strides = match attributes.get("strides") {
                Some(Attribute::Ints(strides)) => strides,
                _ => &default_strides
            };
            let default_dilations = input_tensors[0].shape()[2..].iter().map(|_| 1).collect::<Vec<i64>>();
            let dilations = match attributes.get("dilations") {
                Some(Attribute::Ints(dilations)) => dilations,
                _ => &default_dilations
            };
            let default_pads = input_tensors[0].shape()[2..].iter().map(|_| [0, 0]).flat_map(|x| x).collect::<Vec<i64>>();
            let pads = match attributes.get("pads") {
                Some(Attribute::Ints(pads)) => pads,
                _ => &default_pads
            };
            let group = match attributes.get("group") {
                Some(Attribute::Int(group)) => *group,
                _ => 1
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
            let output = ArrayMultiType::convolution(input_tensors[0], input_tensors[1], bias, auto_pad.as_str(), &dilations, group, &kernel_shape, pads.as_slice(), &strides);
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
        "BatchNormalization" => {
            let epsilon = match attributes.get("epsilon") {
                Some(Attribute::Float(epsilon)) => *epsilon,
                _ => 1e-05
            };
            let momentum = match attributes.get("momentum") {
                Some(Attribute::Float(momentum)) => *momentum,
                _ => 0.9
            };
            let training_mode = match attributes.get("training_mode") {
                Some(Attribute::Int(training_mode)) => *training_mode != 0,
                _ => false
            };
            outputs.insert(node.output[0].clone(), ArrayMultiType::batch_normalization(input_tensors[0], input_tensors[1], input_tensors[2], input_tensors[3], input_tensors[4], epsilon, momentum, training_mode))
        },
        "MaxPool" => {
            let default_auto_pad = "NOTSET".to_string();
            let auto_pad = match attributes.get("auto_pad") {
                Some(Attribute::String(auto_pad)) => auto_pad,
                _ => &default_auto_pad
            };
            let ceil_mode = match attributes.get("ceil_mode") {
                Some(Attribute::Int(ceil_mode)) => *ceil_mode != 0,
                _ => false
            };
            let kernel_shape = match attributes.get("kernel_shape") {
                Some(Attribute::Ints(kernel_shape)) => kernel_shape,
                _ => return Err("Invalid kernel shape")
            };
            let default_strides = input_tensors[0].shape()[2..].iter().map(|_| 1).collect::<Vec<i64>>();
            let strides = match attributes.get("strides") {
                Some(Attribute::Ints(strides)) => strides,
                _ => return Err("Invalid strides")
            };
            let default_dilations = input_tensors[0].shape()[2..].iter().map(|_| 1).collect::<Vec<i64>>();
            let dilations = match attributes.get("dilations") {
                Some(Attribute::Ints(dilations)) => dilations,
                _ => &default_dilations
            };
            let default_pads = input_tensors[0].shape()[2..].iter().map(|_| [0, 0]).flat_map(|x| x).collect::<Vec<i64>>();
            let pads = match attributes.get("pads") {
                Some(Attribute::Ints(pads)) => pads,
                _ => &default_pads
            };
            let storage_order = match attributes.get("storage_order") {
                Some(Attribute::Int(storage_order)) => *storage_order != 0,
                _ => false
            };
            outputs.insert(node.output[0].clone(), ArrayMultiType::max_pool(input_tensors[0], auto_pad.as_str(), ceil_mode, dilations.as_slice(), kernel_shape.as_slice(), pads.as_slice(), storage_order, strides.as_slice()))
        },
        "Split" => {
            let axis = match attributes.get("axis") {
                Some(Attribute::Int(axis)) => *axis as usize,
                _ => 0
            };
            let split = match input_tensors.len() {
                2.. => input_tensors[1].to_vec_usize(),
                _ => match attributes.get("split") {
                    Some(Attribute::Ints(split)) => split.iter().map(|x| *x as usize).collect::<Vec<usize>>(),
                    _ => match attributes.get("num_outputs") {
                        Some(Attribute::Int(num_outputs)) => (0..*num_outputs as usize).map(|n| {
                                let split = input_tensors[0].shape()[axis] / *num_outputs as usize;
                                if n < input_tensors[0].shape()[axis] % *num_outputs as usize {
                                    split + 1
                                } else {
                                    split
                                }
                            }).collect::<Vec<usize>>(),
                        _ => return Err("Invalid num_outputs")
                    }
                }
            };
            
            let output_vec = ArrayMultiType::split(input_tensors[0], split, axis);
            for (index, output) in output_vec.into_iter().enumerate() {
                outputs.insert(node.output[index].clone(), output);
            }
            None
        },
        "ReduceMean" => {
            let keepdims = match attributes.get("keepdims") {
                Some(Attribute::Int(keepdims)) => *keepdims != 0,
                _ => true
            };
            let noop_with_empty_axes = match attributes.get("noop_with_empty_axes") {
                Some(Attribute::Int(noop_with_empty_axes)) => *noop_with_empty_axes != 0,
                _ => false
            };
            let axes = match input_tensors.len() {
                2.. => Some(input_tensors[1].to_vec_i64()),
                _ => match attributes.get("axes") {
                    Some(Attribute::Ints(axes)) => Some(axes.iter().map(|x| *x as i64).collect::<Vec<i64>>()),
                    _ => None
                }
            };
            outputs.insert(node.output[0].clone(), ArrayMultiType::reduce_mean(input_tensors[0], axes, keepdims, noop_with_empty_axes))
        },
        _ => return Err("Operation not supported")        
    };

    Ok(outputs)
}

pub fn execute_graph(graph: &GraphProto, inputs: &mut HashMap<String, ArrayMultiType>, verbose: bool) -> Result<HashMap<String, ArrayMultiType>, &'static str> {
    let shared_inputs = Arc::new((Mutex::new(inputs.clone()), Condvar::new()));
    let mut threads = Vec::new();
    
    for (index, node) in graph.node.iter().enumerate() {
        let node_clone = node.clone();
        let verbose_clone = verbose.clone();
        let shared_inputs_clone = Arc::clone(&shared_inputs);
        threads.push(thread::spawn(move || {
            // Aquiring the lock
            let (lock, cvar) = &*shared_inputs_clone;
            let mut inputs = lock.lock().unwrap();

            if verbose_clone {
                println!("{:?} - {:?}: Start and wait inputs: {:?}", node_clone.name, node_clone.op_type, node_clone.input);
            }
            //Check if all inputs are available, if not relese the lock and wait for the condition variable
            while node_clone.input.iter().any(|input| !inputs.contains_key(input)) {
                inputs = cvar.wait(inputs).unwrap();
            }
            // Get the inputs and relese the lock
            let inputs_clone = inputs.clone();
            drop(inputs);

            let input_shape = node_clone.input.iter().map(|input| inputs_clone.get(input).unwrap().shape()).collect::<Vec<&[usize]>>();

            if verbose_clone {
                println!("{:?} - {:?}: Execute node {:?}", node_clone.name, node_clone.op_type, input_shape);
            }
            // Execute the nod
            let start = Instant::now();
            let node_outputs = execute_node(&node_clone, &inputs_clone).unwrap();
            let duration = start.elapsed();
            if verbose_clone {
                println!("{:?} - {:?}: Execution time: {:?}", node_clone.name, node_clone.op_type, duration);
            }
            
            // Aquiring the lock
            let (lock, cvar) = &*shared_inputs_clone;
            let mut inputs = lock.lock().unwrap();

            // Update the inputs and notify the condition variable
            if verbose {
                println!("{:?} - {:?}: Update inputs: {:?} -> {:?}", node_clone.name, node_clone.op_type, node_outputs.keys(), node_outputs.values().map(|x| x.shape()).collect::<Vec<&[usize]>>());
            }
            inputs.extend(node_outputs.clone());
            cvar.notify_all();
        }));

        if  !verbose {
            print!("Node launched: {}/{}\r", index + 1, graph.node.len());
        }
    }
    println!();

    for (index, thread) in threads.into_iter().enumerate() {
        thread.join().map_err(|_| "Could not join thread")?;
        if !verbose {
            print!("Node terminated: {}/{}\r", index + 1, graph.node.len());
        }
    }
    println!();

    let (lock, _) = &*shared_inputs;
    let inputs = lock.lock().unwrap();
    let outputs = graph.output.iter().map(|output| (output.name.clone(), inputs.get(&output.name).unwrap().clone())).collect::<HashMap<String, ArrayMultiType>>();

    Ok(outputs)
}
