use crate::onnx::{ModelProto, NodeProto};
use std::collections::HashMap;
use ndarray::{Array2, Zip, Array, ArrayBase, Ix1, IxDyn, Dimension, Axis, OwnedRepr, s, Data};
use std::ops::{Add, Div, Sub, Mul};
use num_traits::float::Float;
use std::cmp::{max, min};
use std::iter::FromIterator;
use num_traits::Zero;
use rayon::prelude::*;
use ndarray_parallel::prelude::*;

// Funzione per stampare una lista di tensori
pub fn print_tensors(tensors: Vec<Array2<f32>>) {
    for (index, tensor) in tensors.iter().enumerate() {
        println!("Tensor {}:\n{:?}", index + 1, tensor);
    }
}

// Moltiplicazione di due tensori (element-wise)
pub fn multiply<A>(tensor_a: &ArrayBase<OwnedRepr<A>, IxDyn>, tensor_b: &ArrayBase<OwnedRepr<A>, IxDyn>) -> Array<A, IxDyn>
where
    A: Mul<Output = A> + Clone,
{
    tensor_a * tensor_b
}


// Addizione di due tensori (element-wise)
pub fn add<A>(tensor_a: &ArrayBase<OwnedRepr<A>, IxDyn>, tensor_b: &ArrayBase<OwnedRepr<A>, IxDyn>) -> Array<A, IxDyn>
where
    A: Add<Output = A> + Clone,
{
    tensor_a + tensor_b
}


//Esponenziale di un tensore (element-wise)
pub fn exp<A>(tensor: &ArrayBase<OwnedRepr<A>, IxDyn>) -> Array<A, IxDyn>
where
    A: Float + Sync + Send,
{
    tensor.mapv(|x| x.exp())
}


//Floor di un tensore (element-wise)
pub fn floor<A>(tensor: &ArrayBase<OwnedRepr<A>, IxDyn>) -> Array<A, IxDyn>
where
    A: Float + Sync + Send,
{
    tensor.mapv(|x| x.floor())
}


//Logaritmo naturale di un tensore (element-wise)
pub fn log<A>(tensor: &ArrayBase<OwnedRepr<A>, IxDyn>) -> Array<A, IxDyn>
where
    A: Float + Sync + Send,
{
    tensor.mapv(|x| x.ln())
}


//Operatori logici
// Comparazione elemento per elemento di due tensori, restituisce un tensore di booleani
pub fn greater<A, S>(tensor_a: &ArrayBase<S, IxDyn>, tensor_b: &ArrayBase<S, IxDyn>) -> ArrayBase<S, IxDyn>
where
    A: Sync + Send + PartialOrd,
    S: Data<Elem = A>,
{
    Zip::from(tensor_a)
        .and(tensor_b)
        .par_map_collect(|a, b| a > b)
}

pub fn greater_or_equal<A, S>(tensor_a: &ArrayBase<S, IxDyn>, tensor_b: &ArrayBase<S, IxDyn>) -> ArrayBase<S, IxDyn> 
where
    A: Sync + Send + PartialOrd,
    S: Data<Elem = A>,
{
    Zip::from(tensor_a)
        .and(tensor_b)
        .par_map_collect(|a, b| a>=b)
}

pub fn less<A, S>(tensor_a: &ArrayBase<S, IxDyn>, tensor_b: &ArrayBase<S, IxDyn>) -> ArrayBase<S, IxDyn> 
where
    A: Sync + Send + PartialOrd,
    S: Data<Elem = A>,
{
    Zip::from(tensor_a)
        .and(tensor_b)
        .par_map_collect(|a, b| a<b)
}

pub fn less_or_equal<A, S>(tensor_a: &ArrayBase<S, IxDyn>, tensor_b: &ArrayBase<S, IxDyn>) -> ArrayBase<S, IxDyn> 
where
    A: Sync + Send + PartialOrd,
    S: Data<Elem = A>,
{
    Zip::from(tensor_a)
        .and(tensor_b)
        .par_map_collect(|a, b| a<=b)
}

// Moltiplicazione di matrici per tensori di dimensioni dinamiche
pub fn matmul_tensors<S>(arr1: &ArrayBase<S, IxDyn>, arr2: &ArrayBase<S, IxDyn>) -> ArrayBase<S, IxDyn>
where
    S: Data<Elem = f32>,
{
    arr1.dot(arr2)
}

// Sottrazione di due tensori (element-wise)
pub fn sub_tensors<S>(tensor_a: &ArrayBase<S, IxDyn>, tensor_b: &ArrayBase<S, IxDyn>) -> ArrayBase<S, IxDyn>
where
    S: Data<Elem = f32>,
{
    tensor_a - tensor_b
}

// Divisione di due tensori (element-wise)
pub fn div_tensors<S>(tensor_a: &ArrayBase<S, IxDyn>, tensor_b: &ArrayBase<S, IxDyn>) -> ArrayBase<S, IxDyn>
where
    S: Data<Elem = f32>,
{
    tensor_a / tensor_b
}

// Massimizzazione di due tensori (element-wise)
pub fn max_tensors<S>(tensor_a: &ArrayBase<S, IxDyn>, tensor_b: &ArrayBase<S, IxDyn>) -> ArrayBase<S, IxDyn>
where
    S: Data<Elem = f32>,
{
    Zip::from(tensor_a)
        .and(tensor_b)
        .par_map_collect(|&a, &b| a.max(b))
}

// Minimizzazione di due tensori (element-wise)
pub fn min_tensors<S>(tensor_a: &ArrayBase<S, IxDyn>, tensor_b: &ArrayBase<S, IxDyn>) -> ArrayBase<S, IxDyn>
where
    S: Data<Elem = f32>,
{
    Zip::from(tensor_a)
        .and(tensor_b)
        .par_map_collect(|&a, &b| a.min(b))
}


pub fn global_average_pool<T>(input: &Array<T, IxDyn>) -> Array<T, IxDyn>
where
    T: Clone + Default + std::ops::Add<Output = T> + From<f32> + std::ops::Div<Output = T> + Zero + FromPrimitive,
{
    let input_shape = input.dim();
    if input_shape.ndim() < 3 {
        panic!("Input tensor must have at least 3 dimensions (N x C x D1 x ... x Dn)");
    }

    let spatial_dim_product: f64 = input_shape.slice()[2..]
        .iter()
        .map(|&dim| dim as f64)
        .product();
    let spatial_dim_product: T = T::from_f64(spatial_dim_product).unwrap_or_else(T::zero);

    let mut mean = input.to_owned();
    for axis in (2..input_shape.ndim()).rev() {
        mean = mean.mean_axis(Axis(axis)).unwrap();
    }

    let output_shape = vec![input_shape[0], input_shape[1], 1, 1];  // Per N x C x 1 x 1
    let mut output = Array::default(IxDyn(&output_shape));

    // Itera manualmente sugli elementi di output e mean
    for n in 0..output_shape[0] {
        for c in 0..output_shape[1] {
            output[[n, c, 0, 0]] = mean[[n, c]].clone() / spatial_dim_product.clone();
        }
    }

    output
}

pub fn concat<T>(tensors: Vec<Array<T, IxDyn>>, axis: isize) -> Array<T, IxDyn>
where
    T: Clone,
{
    // Verifica che i tensori non siano vuoti
    if tensors.is_empty() {
        panic!("La lista dei tensori per la concatenazione è vuota");
    }

    // Calcola l'asse effettivo per la concatenazione
    let ndim = tensors[0].ndim();
    let axis = if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    };

    if axis >= ndim {
        panic!("Asse di concatenazione non valido");
    }

    // Verifica che tutti i tensori abbiano la stessa forma, tranne l'asse di concatenazione
    for tensor in &tensors[1..] {
        if tensor.ndim() != ndim {
            panic!("Tutti i tensori devono avere lo stesso numero di dimensioni");
        }
        for (i, dim) in tensor.dim().slice().iter().enumerate() {
            if i != axis && *dim != tensors[0].dim()[i] {
                panic!("Le dimensioni dei tensori devono essere uguali tranne che sull'asse di concatenazione");
            }
        }
    }

    // Concatenazione lungo l'asse specificato
    ndarray::concatenate(Axis(axis), &tensors.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap()
}

// Funzione per eseguire ReLU elementwise
pub fn relu<T>(input: &ArrayBase<OwnedRepr<T>, IxDyn>) -> Array<T, IxDyn>
where
    T: Default + Clone + PartialOrd,  // Rimuovi il vincolo `Ord`
{
    input.mapv(|x| if x > T::default() { x } else { T::default() })
}

fn apply_padding<T>(input: &Array<T, IxDyn>, pads: &[i64]) -> Array<T, IxDyn>
where
    T: Default + Clone,
{
    // Assuming 2D convolution for simplicity. For higher dimensions, this needs to be extended.
    let pad_top = pads[0] as usize;
    let pad_left = pads[1] as usize;
    let pad_bottom = pads[2] as usize;
    let pad_right = pads[3] as usize;

    let mut padded_shape = input.raw_dim();
    padded_shape[2] += pad_top + pad_bottom; // Height dimension
    padded_shape[3] += pad_left + pad_right; // Width dimension

    let mut padded_input = Array::<T, _>::default(padded_shape);

    // Copying the input tensor into the center of the padded tensor
    let slice_s = s![
        ..,
        ..,
        pad_top..(pad_top + input.shape()[2]),
        pad_left..(pad_left + input.shape()[3])
    ];
    padded_input.slice_mut(slice_s).assign(input);

    padded_input
}

// Funzione convolution aggiornata per includere il padding
pub fn convolution<T>(
    input: &Array<T, IxDyn>, // Input tensor X
    weights: &Array<T, IxDyn>, // Weight tensor W
    bias: Option<&Array<T, Ix1>>, // Optional Bias tensor B
    auto_pad: &str, // auto_pad attribute
    dilations: Vec<i64>, // dilations attribute
    group: i64, // group attribute
    kernel_shape: Vec<i64>, // kernel_shape attribute
    pads: Vec<i64>, // pads attribute
    strides: Vec<i64>, // strides attribute
) -> Array<T, IxDyn>
where
    T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy + Default + Clone + Zero,
{
    // Determina il padding necessario
    let input_shape = input.dim();
    let actual_pads = if auto_pad != "NOTSET" {
        let mut pad = vec![0; kernel_shape.len() * 2];  // Due valori di padding per dimensione (inizio e fine)
    
        for (i, &k) in kernel_shape.iter().enumerate() {
            let input_size = input_shape[i + 2] as i64;  // +2 per saltare le prime due dimensioni (N e C)
            let stride = strides[i] as i64;
            let output_size = (input_size + stride - 1) / stride;  // Calcola le dimensioni dell'output
            let total_pad = (output_size - 1) * stride + k as i64 - input_size;  // Calcola il padding totale necessario
            if auto_pad == "SAME_UPPER" {
                pad[i * 2] = total_pad / 2;  // Padding all'inizio
                pad[i * 2 + 1] = total_pad - pad[i * 2];  // Padding alla fine
            } else if auto_pad == "SAME_LOWER" {
                pad[i * 2 + 1] = total_pad / 2;  // Padding alla fine
                pad[i * 2] = total_pad - pad[i * 2 + 1];  // Padding all'inizio
            }
        }
        pad
    } else {
        pads.to_vec()
    };
    

    // Applica il padding al tensore di input
    let padded_input = apply_padding(input, &actual_pads);

    // Calcola le dimensioni del tensore di output
    let input_shape = padded_input.dim();

    // Converte tutti i valori coinvolti in usize prima di eseguire operazioni
    let kernel_height = kernel_shape[0] as usize;
    let kernel_width = kernel_shape[1] as usize;
    let pad_top = actual_pads[0] as usize;
    let pad_left = actual_pads[1] as usize;

    let output_height = ((input_shape[2] - kernel_height + 2 * pad_top) / strides[0] as usize) + 1;
    let output_width = ((input_shape[3] - kernel_width + 2 * pad_left) / strides[1] as usize) + 1;

    let output_shape = vec![input_shape[0], weights.dim()[0], output_height, output_width];
    let mut output = Array::<T, _>::zeros(IxDyn(&output_shape));

    // Converti output in una vista mutabile
    let mut output_view_mut = output.view_mut();

        // Calcola le dimensioni dei gruppi
    let num_groups = group as usize;
    let input_group_size = input_shape[1] / num_groups;
    let weight_group_size = weights.dim()[1] / num_groups;

    // Itera sui gruppi
    for g in 0..num_groups {
        // Calcola gli indici di inizio e fine per il gruppo corrente
        let input_group_start = g * input_group_size;
        let weight_group_start = g * weight_group_size;
        let weight_group_end = weight_group_start + weight_group_size;

        // Utilizza par_iter_mut per iterare parallelamente sulla vista mutabile
        output_view_mut.indexed_iter_mut().for_each(|(idx, output_element)| {
            let (n, m, h, w) = (
                idx[0],
                idx[1] + weight_group_start,  // Aggiusta l'indice in base al gruppo
                idx[2],
                idx[3],
            );

            if m < weight_group_end {
                let mut sum = T::zero();

                for kh in 0..kernel_height {
                    for kw in 0..kernel_width {
                        // Aggiungi la dilatazione all'indice
                        let h_idx = h * strides[0] as usize + kh * dilations[0] as usize;
                        let w_idx = w * strides[1] as usize + kw * dilations[1] as usize;

                        // Assicurati di utilizzare il canale corretto del gruppo per il tensore di input
                        let input_channel = n * input_group_size + input_group_start;

                        // Aggiungi la moltiplicazione elemento per elemento al sum
                        sum = sum + padded_input[[n, input_channel, h_idx, w_idx]] * weights[[m, kh, kw]];
                    }
                }

                // Aggiungi il bias se presente
                if let Some(b) = bias {
                    sum = sum + b[m];
                }

                // Assegna il valore calcolato al tensore di output
                *output_element = sum;
            }
        });
    }

    output

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
            //let result_add = add_tensors(tensor_a, tensor_b);
            let result_sub = sub_tensors(tensor_a.clone(), tensor_b.clone());
            let result_div = div_tensors(tensor_a.clone(), tensor_b.clone());
            let result_max = max_tensors(&tensor_a, &tensor_b);
            let result_min = min_tensors(&tensor_a, &tensor_b);

            // Raccogli i tensori in una lista
            let tensors_to_print = vec![
                tensor_a,
                tensor_b,
                //result_add,
                result_sub,
                result_div,
                result_max,
                result_min,
            ];

            // Stampa tutti i tensori
            print_tensors(tensors_to_print);
        
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
