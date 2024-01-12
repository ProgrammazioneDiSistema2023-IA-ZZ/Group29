use crate::onnx::*;
use ndarray::*;
use std::ops::{Add, Div, Sub, Mul};
use ndarray::Array;
use ndarray::Zip;


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
pub fn add_tensors<A: Add<Output = A> + Clone, D: Dimension>(tensor_a: Array<A, D>, tensor_b: Array<A, D>) -> Array<A, D> {
    // Effettua la somma dei tensori
    tensor_a + tensor_b
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
            let result_add = add_tensors(tensor_a.clone(), tensor_b.clone());
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
                result_add,
                result_sub,
                result_mul,
                result_div,
                result_max,
                result_min,
            ];

            // Stampa tutti i tensori
            print_tensors(tensors_to_print);
        
    }
