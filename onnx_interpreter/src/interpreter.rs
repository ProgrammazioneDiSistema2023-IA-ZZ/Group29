use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use ndarray::{Array, IxDyn};
use tensor_proto::DataType;
use std::sync::Arc;

use crate::onnx::*;
use crate::utils::*;
use crate::attribute::Attribute;
use crate::array::ArrayMultiType;
use rayon::prelude::*;

// Struttura per tenere traccia delle dipendenze.
struct DependencyTracker {
    // Mappa di ogni nodo alle sue dipendenze (nodi dai quali dipende).
    dependencies: HashMap<String, HashSet<String>>,
    // Mappa di ogni nodo ai suoi successori (nodi che dipendono da esso).
    successors: HashMap<String, HashSet<String>>,
}

impl DependencyTracker {
    /// Crea una nuova istanza di DependencyTracker basata sul grafo fornito.
    fn new(graph: &GraphProto) -> Self {
        let mut dependencies = HashMap::new();
        let mut successors = HashMap::new();

        // Itera sui nodi per costruire la mappa delle dipendenze e dei successori.
        for node in &graph.node {
            let node_id = &node.name;

            // Per ogni input del nodo, aggiungi questo nodo ai successori dell'input.
            for input_name in &node.input {
                successors.entry(input_name.clone()).or_insert_with(HashSet::new).insert(node_id.clone());
            }

            // Aggiungi l'elenco degli input del nodo come sue dipendenze.
            dependencies.insert(node_id.clone(), node.input.iter().cloned().collect());
        }

        DependencyTracker {
            dependencies,
            successors,
        }
    }

    // Aggiorna il tracker quando un nodo è stato eseguito.
    fn update(&mut self, executed_node_id: &str) {
        // Rimuovi il nodo dalla lista delle dipendenze di ogni successore.
        if let Some(successors) = self.successors.remove(executed_node_id) {
            for successor_id in successors {
                if let Some(deps) = self.dependencies.get_mut(&successor_id) {
                    deps.remove(executed_node_id);
                }
            }
        }
    }

    // Restituisce l'ID di un nodo eseguibile (senza dipendenze).
    fn get_executable_node(&self) -> Option<String> {
        self.dependencies.iter().find_map(|(node_id, deps)| {
            if deps.is_empty() {
                Some(node_id.clone())
            } else {
                None
            }
        })
    }
}

pub fn execute_node(node: &NodeProto, inputs: &HashMap<String, Arc<ArrayMultiType>>) -> Result<HashMap<String, ArrayMultiType>, &'static str> {
    let mut outputs = HashMap::new();
    let op_type = &node.op_type;

    // Estraiamo i valori da Arc e raccogliamo i riferimenti agli ArrayMultiType
    let input_tensors = node.input.iter().map(|input| {
        Arc::as_ref(inputs.get(input).unwrap()) // Ottiene un riferimento a ArrayMultiType
    }).collect::<Vec<&ArrayMultiType>>();

    let attributes = get_attributes(node)?;
    match op_type.as_str() {
        // Aggiorna le chiamate alle operazioni qui in base al nuovo tipo di input
        "Mul" => outputs.insert(node.output[0].clone(), ArrayMultiType::multiply(&input_tensors[0], &input_tensors[1])),
        "Add" => outputs.insert(node.output[0].clone(), ArrayMultiType::add(&input_tensors[0], &input_tensors[1])),
        "Greater" => outputs.insert(node.output[0].clone(), ArrayMultiType::greater(&input_tensors[0], &input_tensors[1])),
        "GreaterOrEqual" => outputs.insert(node.output[0].clone(), ArrayMultiType::greater_or_equal(&input_tensors[0], &input_tensors[1])),
        "Less" => outputs.insert(node.output[0].clone(), ArrayMultiType::less(&input_tensors[0], &input_tensors[1])),
        "LessOrEqual" => outputs.insert(node.output[0].clone(), ArrayMultiType::less_or_equal(&input_tensors[0], &input_tensors[1])),
        "MatMul" => outputs.insert(node.output[0].clone(), ArrayMultiType::matmul(&input_tensors[0], &input_tensors[1])),
        "GlobalAveragePool" => outputs.insert(node.output[0].clone(), ArrayMultiType::global_average_pool(&input_tensors[0])),
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
            let output = ArrayMultiType::convolution(input_tensors[0], input_tensors[1], bias, &auto_pad, &dilations, group, &kernel_shape, &pads, &strides);
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

fn convert_tensor_to_array(tensor: &TensorProto) -> Result<ArrayMultiType, &'static str> {

    let shape = IxDyn(&tensor.dims.iter().map(|&d| d as usize).collect::<Vec<_>>());

    match DataType::try_from(tensor.data_type).map_err(|_| "Invalid data type")? {
        DataType::Float => {
            let data = tensor.float_data.clone();
            let array = Array::from_shape_vec(shape, data).map_err(|_| "Failed to create ndarray")?;
            Ok(ArrayMultiType::FLOAT(array))
        },
        DataType::Int32 => {
            let data = tensor.int32_data.clone();
            let array = Array::from_shape_vec(shape, data).map_err(|_| "Failed to create ndarray")?;
            Ok(ArrayMultiType::INT32(array))
        },
        // Aggiungi altri casi per diversi tipi di dati...
        _ => Err("Unsupported tensor data type")
    }
}

fn prepare_node_inputs(node: &NodeProto, results: &HashMap<String, Arc<ArrayMultiType>>) -> Result<HashMap<String, Arc<ArrayMultiType>>, &'static str> {
    let mut inputs = HashMap::new();

    for input_name in &node.input {
        if let Some(input_array) = results.get(input_name) {
            inputs.insert(input_name.clone(), Arc::clone(input_array));
        } else {
            return Err("Missing input for node");
        }
    }

    Ok(inputs)
}

pub fn execute_graph(graph: &GraphProto) -> Result<HashMap<String, Arc<ArrayMultiType>>, &'static str> {
    let mut tracker = DependencyTracker::new(graph);
    let mut results = HashMap::new();

    // Inizializza `results` con gli `initializer`, avvolgendoli in `Arc`
    for initializer in &graph.initializer {
        let array = convert_tensor_to_array(initializer)?;
        results.insert(initializer.name.clone(), Arc::new(array));
    }

    // Continua ad eseguire i nodi finché ci sono nodi eseguibili
    while let Some(node_id) = tracker.get_executable_node() {
        let node = graph.node.iter()
            .find(|n| n.name == node_id)
            .ok_or("Node not found")?;

        let inputs = prepare_node_inputs(node, &results)?;

        let node_outputs = execute_node(node, &inputs)?;

        // Aggiorna i risultati con gli output del nodo, avvolgendoli in `Arc`
        for (output_name, output_value) in node_outputs {
            results.insert(output_name, Arc::new(output_value));
        }

        tracker.update(&node_id);
    }

    Ok(results)
}