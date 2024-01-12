use crate::onnx::{ModelProto, NodeProto, TensorProto};
use std::collections::HashMap;
use ndarray::{Array2, Zip, Array, ArrayBase, Data, IxDyn, Dimension, linalg::Dot};
use std::ops::{Add, Div, Sub, Mul};

// Funzione per stampare una lista di tensori
pub fn print_tensors(tensors: Vec<Array2<f32>>) {
    for (index, tensor) in tensors.iter().enumerate() {
        println!("Tensor {}:\n{:?}", index + 1, tensor);
    }
}

// Moltiplicazione di due tensori
pub fn multiply_tensors<A: Mul<Output = A> + Clone, D: Dimension>(tensor_a: Array<A, D>, tensor_b: Array<A, D>) -> Array<A, D> {
    // Effettua la moltiplicazione dei tensori
    tensor_a * tensor_b
}

// Addizione di due tensori
pub fn add_tensors<A: Add<Output = A> + Clone, D: Dimension>(tensor_a: &Array<A, D>, tensor_b: &Array<A, D>) -> Array<A, D> {
    // Effettua la somma dei tensori
    tensor_a + tensor_b
}

pub fn matmul_tensors(arr1: &Array2<f32>, arr2: &Array2<f32>) -> Array2<f32> {
    arr1.dot(arr2)
}

// Sottrazione di due tensori
pub fn sub_tensors<T>(tensor_a: Array2<T>, tensor_b: Array2<T>) -> Array2<T>
where
    T: Sub<Output = T> + Clone,
{
    tensor_a - &tensor_b
}

// Moltiplicazione di due tensori (simile a quanto già implementato)
pub fn mul_tensors<T>(tensor_a: Array2<T>, tensor_b: Array2<T>) -> Array2<T>
where
    T: Mul<Output = T> + Clone,
{
    tensor_a * &tensor_b
}

// Divisione di due tensori
pub fn div_tensors<T>(tensor_a: Array2<T>, tensor_b: Array2<T>) -> Array2<T>
where
    T: Div<Output = T> + Clone,
{
    tensor_a / &tensor_b
}

// Funzione per eseguire la massimizzazione di due tensori
pub fn max_tensors(tensor_a: &Array2<f32>, tensor_b: &Array2<f32>) -> Array2<f32> {
    // Esegui la massimizzazione dei tensori
    let result_tensor = Zip::from(tensor_a)
        .and(tensor_b)
        .map_collect(|&a, &b| a.max(b));

    // Restituisci il risultato
    result_tensor
}

// Funzione per eseguire la minimizzazione di due tensori
pub fn min_tensors(tensor_a: &Array2<f32>, tensor_b: &Array2<f32>) -> Array2<f32> {
    // Esegui la minimizzazione dei tensori
    let result_tensor = Zip::from(tensor_a)
        .and(tensor_b)
        .map_collect(|&a, &b| a.min(b));

    // Restituisci il risultato
    result_tensor
}


// Funzione principale per eseguire le operazioni specifiche sul modello ONNX
pub fn perform_operations(model: ModelProto) {
    // Carica il modello ONNX utilizzando la funzione dal modulo onnx_handler
    

            if let Some(graph) = model.graph.as_ref() {
                for node in &graph.node {
                    println!("Node: {:?}", node);
                }
            }

            // Esempio di tensori
            let tensor_a = Array2::from_shape_fn((2, 2), |(i, j)| (i + j) as f32);
            let tensor_b = Array2::from_shape_fn((2, 2), |(i, j)| (i * j) as f32);

            let result_multiply = multiply_tensors(tensor_a.clone(), tensor_b.clone());
            //let result_add = add_tensors(tensor_a, tensor_b);
            let result_sub = sub_tensors(tensor_a.clone(), tensor_b.clone());
            let result_mul = mul_tensors(tensor_a.clone(), tensor_b.clone());
            let result_div = div_tensors(tensor_a.clone(), tensor_b.clone());
            let result_max = max_tensors(&tensor_a, &tensor_b);
            let result_min = min_tensors(&tensor_a, &tensor_b);

            // Raccogli i tensori in una lista
            let tensors_to_print = vec![
                tensor_a,
                tensor_b,
                result_multiply,
                //result_add,
                result_sub,
                result_mul,
                result_div,
                result_max,
                result_min,
            ];

            // Stampa tutti i tensori
            print_tensors(tensors_to_print);
        
    }



pub fn execute_onnx(
    node: &NodeProto,
    input_tensors: &HashMap<String, Array<f32, IxDyn>>,
    output_tensors: &mut HashMap<String, Array<f32, IxDyn>>,
) {
    // Estrai informazioni dal nodo ONNX
    let op_type = &node.op_type;
    let input_names = &node.input;
    let output_names = &node.output;
    let attributes = &node.attribute;

    // Gestisci diverse operazioni in base al tipo di operatore (op_type)
    match op_type.as_str() {
        // Aggiungi casi per ciascun tipo di operatore supportato
        "Add" => {
            // Esempio: Somma due tensori
            let input_a = input_tensors.get(&input_names[0]).unwrap();
            let input_b = input_tensors.get(&input_names[1]).unwrap();
            let output = input_a + input_b;
            output_tensors.insert(output_names[0].clone(), output);
        }
        // Aggiungi altri casi per altri operatori supportati
        // ...

        // Gestisci i casi non supportati
        _ => {
            eprintln!("Unsupported operator type: {}", op_type);
            // Puoi gestire l'operatore sconosciuto in modo specifico o ignorarlo
        }
    }
}


pub fn inference(
    model: ModelProto,
    input_tensors: HashMap<String, Array<f32, IxDyn>>,
) -> HashMap<String, Array<f32, IxDyn>> {
    let mut output_tensors: HashMap<String, Array<f32, IxDyn>> = HashMap::new();

    // Accedi direttamente al campo `graph`
    if let Some(graph) = model.graph {
        // Estrai il grafo e gli input
        let nodes = graph.node;
        //let initializers = graph.initializer;

        // Prepara i tensori di input
        let mut all_tensors: HashMap<String, Array<f32, IxDyn>> = HashMap::new();
        all_tensors.extend(input_tensors);

        // Aggiungi gli initializer ai tensori di input
        // for initializer in initializers {
        //     let name = initializer.name;
        //     let array = convert_tensor_proto_to_array(&initializer);
        //     all_tensors.insert(name, array);
        // }

        // Itera sui nodi del grafo e esegui l'inferenza
        for node in nodes {
            execute_onnx(&node, &all_tensors, &mut output_tensors);
        }
    }

    // Restituisci i risultati
    output_tensors
}

// fn convert_tensor_proto_to_array(tensor_proto: &TensorProto) -> Array<f32, IxDyn> {
//     // Estrai i campi necessari da TensorProto
//     let dims: Vec<usize> = tensor_proto.dims.iter().map(|&d| d as usize).collect();
    
//     // Converte i dati in un Array<f32, IxDyn>
//     let array = match tensor_proto.data_type {
//         DataType::FLOAT => {
//             let tensor_data = tensor_proto.float_data.as_ref().unwrap_or(&vec![]);
//             Array::from_shape_vec(dims.into(), tensor_data.clone()).unwrap()
//         }
//         // Aggiungi altri casi per gli altri tipi di dato supportati
//         _ => {
//             eprintln!("Unsupported data type: {:?}", tensor_proto.data_type);
//             Array::from_shape_vec(IxDyn(&[]), vec![]).unwrap()
//         }
//     };

//     array
// }
