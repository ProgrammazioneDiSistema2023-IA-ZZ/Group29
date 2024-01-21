use crate::onnx::{ModelProto, NodeProto};
use std::collections::HashMap;
use ndarray::{Array2, ArrayD, Zip, Array, Array1, ArrayBase, Ix1, IxDyn, Ix2, Dimension, Axis, OwnedRepr, s, Data, DataMut, ScalarOperand, ArrayViewD, SliceInfo, SliceInfoElem, Slice};
use std::ops::{Add, Sub, Mul};
use num_traits::float::Float;
use std::iter::FromIterator;
use std::cmp;
use std::convert::{TryFrom, TryInto};
use num_traits::{Zero, FromPrimitive, Bounded};
use rayon::prelude::*;
use ndarray_parallel::prelude::*;

// use crate::AutoPad::*;
enum AutoPad {NotSet, SameUpper, SameLower, Valid}
enum Error {AutoPadError, KernelShapeError, DilationError, PadsError, StridesError}

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

// Sottrazione di due tensori (element-wise)
pub fn sub_tensors<A, S>(tensor_a: &ArrayBase<S, IxDyn>, tensor_b: &ArrayBase<S, IxDyn>) -> ArrayD<A>
where
    A: Sub<Output = A> + Clone + Default,
    S: Data<Elem = A>,
{
    // Broadcasting dei tensori per allineare le loro dimensioni
    let tensor_a_broadcast = tensor_a.broadcast(tensor_b.raw_dim()).expect("Shape mismatch for broadcasting");
    let tensor_b_broadcast = tensor_b.broadcast(tensor_a.raw_dim()).expect("Shape mismatch for broadcasting");

    // Sottrazione elemento per elemento utilizzando Zip e map_collect
    Zip::from(&tensor_a_broadcast)
        .and(&tensor_b_broadcast)
        .map_collect(|a, b| a.clone() - b.clone())
}

// Divisione di due tensori (element-wise)
pub fn div_tensors_generic<A, S>(tensor_a: &ArrayBase<S, IxDyn>, tensor_b: &ArrayBase<S, IxDyn>) -> ArrayD<A>
where
    A: std::ops::Div<Output = A> + Clone + Default,
    S: Data<Elem = A>,
{
    // Broadcasting dei tensori per allineare le loro dimensioni
    let tensor_a_broadcast = tensor_a.broadcast(tensor_b.raw_dim()).expect("Shape mismatch for broadcasting");
    let tensor_b_broadcast = tensor_b.broadcast(tensor_a.raw_dim()).expect("Shape mismatch for broadcasting");

    // Divisione elemento per elemento utilizzando Zip e map_collect
    Zip::from(&tensor_a_broadcast)
        .and(&tensor_b_broadcast)
        .map_collect(|a, b| a.clone() / b.clone())
}



// Massimizzazione di due tensori (element-wise)
// Funzione ausiliaria per calcolare la forma di broadcasting
fn compute_broadcast_shape(shapes: Vec<Vec<usize>>) -> Vec<usize> {
    // Trova la lunghezza massima tra le forme
    let max_len = shapes.iter().map(|s| s.len()).max().unwrap_or(0);

    // Calcola la forma di broadcasting
    (0..max_len).rev().map(|i| {
        shapes.iter()
            .filter_map(|s| s.get(s.len().checked_sub(i + 1).unwrap_or(0)))
            .max()
            .cloned()
            .unwrap_or(1)
    }).collect::<Vec<_>>().into_iter().rev().collect()
}

// Implementazione generica per l'operazione max tra più tensori
pub fn max_tensors_generic<A, S>(tensors: &[ArrayBase<S, IxDyn>]) -> ArrayD<A>
where
    A: Clone + Default + PartialOrd,
    S: Data<Elem = A>,
{
    if tensors.is_empty() {
        panic!("At least one tensor is required for max operation");
    }

    // Calcola la forma di broadcasting
    let shapes: Vec<_> = tensors.iter().map(|t| t.shape().to_vec()).collect();
    let broadcast_shape = compute_broadcast_shape(shapes);
    let broadcast_dim = IxDyn(&broadcast_shape);

    // Inizializza il tensore risultante con il primo tensore, dopo aver applicato il broadcasting
    let mut result = tensors[0].broadcast(broadcast_dim.clone()).unwrap().to_owned();

    // Applica l'operazione max elemento per elemento
    for tensor in &tensors[1..] {
        let broadcasted_tensor = tensor.broadcast(broadcast_dim.clone()).unwrap();
        result.zip_mut_with(&broadcasted_tensor, |r, t| {
            if t > r {
                *r = t.clone();
            }
        });
    }

    result
}


// Implementazione generica per l'operazione min tra più tensori
pub fn min_tensors_generic<A, S>(tensors: &[ArrayBase<S, IxDyn>]) -> ArrayD<A>
where
    A: Clone + Default + PartialOrd,
    S: Data<Elem = A>,
{
    if tensors.is_empty() {
        panic!("At least one tensor is required for min operation");
    }

    // Calcola la forma di broadcasting
    let shapes: Vec<_> = tensors.iter().map(|t| t.shape().to_vec()).collect();
    let broadcast_shape = compute_broadcast_shape(shapes);
    let broadcast_dim = IxDyn(&broadcast_shape);

    // Inizializza il tensore risultante con il primo tensore, dopo aver applicato il broadcasting
    let mut result = tensors[0].broadcast(broadcast_dim.clone()).unwrap().to_owned();

    // Applica l'operazione min elemento per elemento
    for tensor in &tensors[1..] {
        let broadcasted_tensor = tensor.broadcast(broadcast_dim.clone()).unwrap();
        result.zip_mut_with(&broadcasted_tensor, |r, t| {
            if t < r {
                *r = t.clone();
            }
        });
    }

    result
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
pub fn greater<A, S>(tensor_a: &ArrayBase<S, IxDyn>, tensor_b: &ArrayBase<S, IxDyn>) -> ArrayBase<OwnedRepr<bool>, IxDyn>
where
    A: Sync + Send + PartialOrd,
    S: Data<Elem = A>,
{
    Zip::from(tensor_a)
        .and(tensor_b)
        .par_map_collect(|a, b| a > b)
}

pub fn greater_or_equal<A, S>(tensor_a: &ArrayBase<S, IxDyn>, tensor_b: &ArrayBase<S, IxDyn>) -> ArrayBase<OwnedRepr<bool>, IxDyn>
where
    A: Sync + Send + PartialOrd,
    S: Data<Elem = A>,
{
    Zip::from(tensor_a)
        .and(tensor_b)
        .par_map_collect(|a, b| a>=b)
}

pub fn less<A, S>(tensor_a: &ArrayBase<S, IxDyn>, tensor_b: &ArrayBase<S, IxDyn>) -> ArrayBase<OwnedRepr<bool>, IxDyn>
where
    A: Sync + Send + PartialOrd,
    S: Data<Elem = A>,
{
    Zip::from(tensor_a)
        .and(tensor_b)
        .par_map_collect(|a, b| a<b)
}

pub fn less_or_equal<A, S>(tensor_a: &ArrayBase<S, IxDyn>, tensor_b: &ArrayBase<S, IxDyn>) -> ArrayBase<OwnedRepr<bool>, IxDyn>
where
    A: Sync + Send + PartialOrd,
    S: Data<Elem = A>,
{
    Zip::from(tensor_a)
        .and(tensor_b)
        .par_map_collect(|a, b| a<=b)
}

pub fn equal<A, S>(tensor_a: &ArrayBase<S, IxDyn>, tensor_b: &ArrayBase<S, IxDyn>) -> ArrayBase<OwnedRepr<bool>, IxDyn>
where
    A: Sync + Send + PartialEq,
    S: ndarray::Data<Elem = A>,
{
    Zip::from(tensor_a)
        .and(tensor_b)
        .par_map_collect(|a, b| a == b)
}

pub fn not(tensor: &ArrayBase<OwnedRepr<bool>, IxDyn>) -> ArrayBase<OwnedRepr<bool>, IxDyn> {
    Zip::from(tensor)
        .par_map_collect(|&x| !x)
}

// Moltiplicazione di matrici per tensori di dimensioni dinamiche
pub fn matmul<S>(
    tensor_a: &ArrayBase<S, IxDyn>,
    tensor_b: &ArrayBase<S, IxDyn>,
) -> Result<ArrayBase<OwnedRepr<f32>, IxDyn>, &'static str>
where
    S: Data<Elem = f32>,
{   
    // Trasforma i tensori in Array2, che rappresenta una matrice 2D
    let tensor_1 = tensor_a.view().into_dimensionality::<ndarray::Ix2>().map_err(|_| "Tensor A with wrong dimensionality")?;
    let tensor_2 = tensor_b.view().into_dimensionality::<ndarray::Ix2>().map_err(|_| "Tensor B with wrong dimensionality")?;
    // Verifica che le dimensioni delle matrici siano compatibili per la moltiplicazione
    if tensor_1.ncols() != tensor_2.nrows() {
        return Err("Le dimensioni delle matrici non sono compatibili per la moltiplicazione.");
    }

    // Esegui la moltiplicazione di matrici
    Ok(tensor_1.dot(&tensor_2).into_dyn())
}

// Funzione per trovare gli indici degli elementi non zero di un tensore
pub fn non_zero<S, A>(tensor: &ArrayBase<S, IxDyn>) -> ArrayD<i64>
where
    S: Data<Elem = A>,
    A: PartialEq + num_traits::Zero + Copy,
{
    let mut non_zero_indices = Vec::new();

    // Itera su tutti gli elementi del tensore
    for (index, &item) in tensor.indexed_iter() {
        if item != A::zero() {
            // Converti gli indici in i64 e aggiungili al vettore
            let idx_i64: Vec<i64> = index.slice().iter().map(|&x| x as i64).collect();
            non_zero_indices.push(idx_i64);
        }
    }

    // Converti il vettore di vettori in un ArrayD
    let shape = IxDyn(&[non_zero_indices.len(), tensor.ndim()]);
    ArrayD::from_shape_vec(shape, non_zero_indices.into_iter().flatten().collect()).unwrap()
}

// Funzione per generare un tensore di una forma specificata con un valore costante
pub fn constant_of_shape(shape: &Array1<i64>, value: Option<&ArrayD<f32>>) -> ArrayD<f32> {
    let shape_usize: Vec<usize> = shape.iter().map(|&dim| usize::try_from(dim).unwrap()).collect();
    let default_value = Array::from_elem(shape_usize.as_slice(), 0.0);

    match value {
        Some(val) => {
            if val.len() == 1 {
                let constant_value = val[[0]];
                Array::from_elem(shape_usize.as_slice(), constant_value)
            } else {
                panic!("Value tensor must have exactly one element.");
            }
        },
        None => default_value,
    }
}

pub fn squeeze<A>(data: &ArrayBase<impl Data<Elem = A>, IxDyn>, axes: Option<&[i64]>) -> ArrayBase<OwnedRepr<A>, IxDyn>
where
    A: Clone,
{
    let mut squeezed = data.view();

    match axes {
        Some(axes) => {
            let axes: Vec<usize> = axes.iter()
                .map(|&axis| if axis < 0 { (data.ndim() as i64 + axis) as usize } else { axis as usize })
                .collect();

            for &axis in axes.iter().rev() {
                if data.shape()[axis] == 1 {
                    squeezed = squeezed.index_axis_move(Axis(axis), 0);
                }
            }
        }
        None => {
            for axis in (0..data.ndim()).rev() {
                if data.shape()[axis] == 1 {
                    squeezed = squeezed.index_axis_move(Axis(axis), 0);
                }
            }
        }
    }

    // Invece di convertire in un array di proprietà, restituire la vista risultante
    squeezed.to_owned()
}

pub fn unsqueeze<A>(data: &Array<A, IxDyn>, axes: &[i64]) -> ArrayD<A>
where
    A: Clone,
{
    let mut expanded_data = data.view();
    let mut added_axes = 0usize; // Usa usize per added_axes

    for &axis in axes.iter() {
        // Calcola l'indice assoluto tenendo conto degli assi già aggiunti
        let abs_axis = if axis < 0 {
            (data.ndim() as i64 + axis + 1 + added_axes as i64) as usize
        } else {
            (axis as usize) + added_axes
        };

        // Inserisci la nuova dimensione
        expanded_data = expanded_data.insert_axis(Axis(abs_axis));
        added_axes += 1;
    }

    expanded_data.to_owned()
}

pub fn softmax<A>(input: &Array<A, IxDyn>, axis: isize) -> ArrayD<A>
where
    A: Float,
{
    let axis = if axis < 0 {
        // Converti l'asse negativo in un indice positivo
        input.ndim() as isize + axis
    } else {
        axis
    } as usize;

    // Calcola l'esponenziale di ciascun elemento nel tensore
    let exp: ArrayD<A> = input.mapv(|a| a.exp());

    // Somma tutti gli esponenziali lungo l'asse specificato
    let sum_exp = exp.sum_axis(Axis(axis));

    // Crea una variabile intermedia per conservare il risultato di insert_axis
    let sum_exp_with_axis = sum_exp.insert_axis(Axis(axis));
    
    // Effettua il broadcast della somma lungo l'asse per allineare le dimensioni
    let sum_exp_broadcast = sum_exp_with_axis.broadcast(exp.dim()).unwrap();

    // Calcola il Softmax dividendo ciascun esponenziale per la somma degli esponenziali lungo l'asse
    exp / sum_exp_broadcast
}

pub fn shape<S>(data: &ArrayBase<S, IxDyn>, start: Option<isize>, end: Option<isize>) -> Vec<i64>
where
    S: Data,
{
    let rank = data.ndim() as isize;
    let start_idx = start.unwrap_or(0).clamp(-rank, rank - 1);
    let end_idx = end.unwrap_or(rank).clamp(-rank, rank);

    // Ajusta gli indici in caso di valori negativi
    let start_idx = if start_idx >= 0 { start_idx as usize } else { (rank + start_idx) as usize };
    let end_idx = if end_idx >= 0 { end_idx as usize } else { (rank + end_idx) as usize };

    data.shape()[start_idx..end_idx].iter().map(|&d| d as i64).collect()
}

pub fn reshape<S>(data: &ArrayBase<S, IxDyn>, new_shape: &[i64]) -> ArrayBase<OwnedRepr<f32>, IxDyn>
where
    S: Data<Elem = f32>,
{
    let mut new_shape_usize: Vec<usize> = Vec::with_capacity(new_shape.len());
    let mut negative_one_index: Option<usize> = None;
    let mut elements_count = 1;

    for (i, &dim) in new_shape.iter().enumerate() {
        match dim {
            -1 => {
                if negative_one_index.is_some() {
                    panic!("Only one dimension can be -1");
                }
                negative_one_index = Some(i);
                new_shape_usize.push(0); // Placeholder value
            }
            0 => {
                new_shape_usize.push(data.shape()[i]);
                elements_count *= data.shape()[i];
            }
            _ => {
                let dim_usize = usize::try_from(dim).expect("Negative dimensions are not allowed");
                new_shape_usize.push(dim_usize);
                elements_count *= dim_usize;
            }
        }
    }

    if let Some(idx) = negative_one_index {
        let original_elements_count: usize = data.iter().count();
        if original_elements_count % elements_count != 0 {
            panic!("Invalid shape for Reshape");
        }
        new_shape_usize[idx] = original_elements_count / elements_count;
    }

    let reshaped = data.view().into_shape(new_shape_usize).expect("Invalid shape for Reshape").to_owned();
    reshaped.into_dyn()
}

pub fn expand<S>(input: &ArrayBase<S, IxDyn>, new_shape: &[i64]) -> ArrayBase<OwnedRepr<f32>, IxDyn>
where
    S: Data<Elem = f32>,
{
    let input_shape = input.shape();
    let mut expanded_shape = Vec::new();

    // Calcola la forma espansa tenendo conto del broadcasting
    for (input_dim, &expand_dim) in input_shape.iter().rev().zip(new_shape.iter().rev()) {
        match expand_dim {
            1 => expanded_shape.push(*input_dim),
            _ => expanded_shape.push(expand_dim as usize),
        }
    }

    expanded_shape.reverse();

    // Crea un nuovo tensore con la forma espansa e copia i dati secondo le regole del broadcasting
    let expanded = ArrayBase::from_elem(expanded_shape, 0.0f32);
    let mut expanded = expanded.into_dyn();
    Zip::from(&mut expanded)
        .and_broadcast(input)
        .for_each(|exp, &inp| *exp = inp);

    expanded
}

pub fn clip<S, A>(
    input: &ArrayBase<S, IxDyn>,
    min: Option<A>,
    max: Option<A>,
) -> ArrayBase<OwnedRepr<A>, IxDyn>
where
    S: Data<Elem = A>,
    A: PartialOrd + Copy + FromPrimitive,
{
    let min_val = min.unwrap_or_else(|| A::from_f32(std::f32::MIN).unwrap());
    let max_val = max.unwrap_or_else(|| A::from_f32(std::f32::MAX).unwrap());

    input.mapv(|x| {
        if x < min_val {
            min_val
        } else if x > max_val {
            max_val
        } else {
            x
        }
    })
}

pub fn scatter_elements<S, A>(
    data: &ArrayBase<S, IxDyn>,
    indices: &ArrayBase<OwnedRepr<i64>, IxDyn>,
    updates: &ArrayBase<S, IxDyn>,
    axis: Option<usize>,
    reduction: Option<&str>,
) -> ArrayBase<OwnedRepr<A>, IxDyn>
where
    S: Data<Elem = A>,
    A: Default + Clone + PartialOrd + std::ops::AddAssign + std::ops::MulAssign,
{
    let mut output = data.to_owned();
    let axis = axis.unwrap_or(0);
    let reduction = reduction.unwrap_or("none");

    for ((idx, update), &index) in updates.indexed_iter().zip(indices.iter()) {
        let mut target_idx = idx.clone();
        target_idx.slice_mut()[axis] = cmp::min(index as usize, data.shape()[axis] - 1);

        match reduction {
            "add" => output[target_idx] += update.clone(),
            "mul" => output[target_idx] *= update.clone(),
            "none" | _ => output[target_idx] = update.clone(),
        }
    }

    output
}

pub fn group_normalization<A>(input: &ArrayBase<OwnedRepr<A>, IxDyn>, num_groups: usize, epsilon: A) -> ArrayBase<OwnedRepr<A>, IxDyn>
where
    A: Float + From<f32> + std::iter::Sum + Default + FromPrimitive,
{
    let (batch_size, channels) = (input.shape()[0], input.shape()[1]);
    let group_size = channels / num_groups;

    let mut output = input.to_owned();

    for b in 0..batch_size {
        for g in 0..num_groups {
            let start_channel = g * group_size;
            let end_channel = start_channel + group_size;

            let slice = s![b, start_channel..end_channel, ..];
            let group_data = input.slice(slice);

            let mean = group_data.mean_axis(Axis(1)).unwrap();
            let variance = group_data.var_axis(Axis(1), A::zero());

            let slice_mut = s![b, start_channel..end_channel, ..];
            Zip::from(output.slice_mut(slice_mut))
                .and(group_data)
                .and_broadcast(&mean)
                .and_broadcast(&variance)
                .for_each(|o, &d, &m, &v| {
                    *o = (d - m) / (v + epsilon).sqrt();
                });
        }
    }

    output
}

pub fn layer_normalization(
    input: &ArrayBase<impl Data<Elem = f32>, IxDyn>,
    scale: &ArrayBase<impl Data<Elem = f32>, IxDyn>,
    bias: &ArrayBase<impl Data<Elem = f32>, IxDyn>,
    axis: isize,
    epsilon: f32,
) -> ArrayBase<OwnedRepr<f32>, IxDyn> {
    let axis = if axis < 0 {
        input.ndim() as isize + axis
    } else {
        axis
    } as usize;

    // Calcolo della media e della varianza lungo l'asse specificato
    let mean = input.mean_axis(Axis(axis)).expect("Failed to compute mean");
    let variance = input.var_axis(Axis(axis), 0.0);

    // Normalizzazione, ridimensionamento e spostamento
    let mut output = ArrayBase::zeros(input.raw_dim());
    Zip::from(&mut output)
        .and(input)
        .and_broadcast(&mean)
        .and_broadcast(&variance)
        .and_broadcast(scale)
        .and_broadcast(bias)
        .for_each(|o, &i, &m, &v, &s, &b| {
            *o = s * ((i - m) / (v + epsilon).sqrt()) + b;
        });

    output
}


pub fn batch_normalization(
    x: &ArrayBase<OwnedRepr<f32>, IxDyn>, 
    scale: &ArrayBase<OwnedRepr<f32>, IxDyn>,
    b: &ArrayBase<OwnedRepr<f32>, IxDyn>,
    input_mean: &ArrayBase<OwnedRepr<f32>, IxDyn>,
    input_var: &ArrayBase<OwnedRepr<f32>, IxDyn>,
    epsilon: f32,
    momentum: f32,
    training_mode: bool
) -> (ArrayBase<OwnedRepr<f32>, IxDyn>, Option<ArrayBase<OwnedRepr<f32>, IxDyn>>, Option<ArrayBase<OwnedRepr<f32>, IxDyn>>)
{
    let sqrt_var = (&*input_var + epsilon).mapv(|var| var.sqrt());
    let norm = (x - input_mean) / &sqrt_var;
    let y = norm * scale + b;

    if training_mode {
        let current_mean = x.mean_axis(Axis(0)).unwrap();
        let current_var = x.var_axis(Axis(0), 0.0);

        let updated_mean = input_mean * momentum + &current_mean * (1.0 - momentum);
        let updated_var = input_var * momentum + &current_var * (1.0 - momentum);

        (y, Some(updated_mean), Some(updated_var))
    } else {
        (y, None, None)
    }
}


// Funzione Slice
pub fn slice_tensor(
    tensor: &ArrayD<f32>,
    starts: &[isize],
    ends: &[isize],
    axes: Option<&[isize]>,
    steps: Option<&[isize]>,
) -> ArrayD<f32> {
    let ndim = tensor.ndim();

    // Imposta gli assi e i passi se non specificati
    let default_axes: Vec<_> = (0..ndim as isize).collect();
    let axes = axes.unwrap_or(&default_axes);
    let default_steps: Vec<_> = vec![1; starts.len()];
    let steps = steps.unwrap_or(&default_steps);

    // (Prossimi passaggi: gestione di starts, ends e slicing effettivo)

    let mut slice_info_vec = Vec::new();

for (&axis, (&start, &end)) in axes.iter().zip(starts.iter().zip(ends.iter())) {
    let dim = tensor.shape()[axis as usize];

    // Calcolo degli indici effettivi
    let start = if start < 0 { start + dim as isize } else { start };
    let end = if end < 0 { end + dim as isize } else { end };

    // Limitazione degli indici
    let start = start.clamp(0, dim as isize);
    let end = end.clamp(0, dim as isize);

    // Costruzione della slice per ogni asse
    slice_info_vec.push(Slice::new(start, Some(end), steps[axis as usize]));
}

        let slice_info_elems = slice_info_vec.into_iter().map(SliceInfoElem::from).collect::<Vec<_>>();

        // Utilizzo di un blocco unsafe per creare SliceInfo
        let slice_info = unsafe {
            SliceInfo::<_, IxDyn, IxDyn>::new(slice_info_elems).unwrap()
        };

        tensor.slice(slice_info.as_ref()).to_owned()

}

pub fn reduce_sum<S, A>(
    data: &ArrayBase<S, IxDyn>,
    axes: Option<&[i64]>,
    keepdims: bool,
    noop_with_empty_axes: bool,
) -> ArrayD<A>
where
    S: Data<Elem = A>,
    A: Clone + Zero + Add<Output = A> + FromPrimitive,
{
    let data_shape = data.shape();
    let rank = data_shape.len();

    // Convert axes to usize, handling negative values
    let axes: Vec<usize> = match axes {
        Some(ax) => ax.iter()
            .map(|&a| if a < 0 { (rank as i64 + a) as usize } else { a as usize })
            .collect(),
        None => (0..rank).collect(),
    };

    // Return the input tensor if noop_with_empty_axes is true and axes is empty
    if axes.is_empty() && noop_with_empty_axes {
        return data.to_owned();
    }

    // Calculate the sum
    let mut result = data.to_owned();
    for &axis in &axes {
        result = result.sum_axis(Axis(axis));
    }

    // Handle keepdims
    if keepdims {
        let mut shape = data_shape.to_vec();
        for &axis in &axes {
            shape[axis] = 1;
        }
        result = result.into_shape(IxDyn(&shape)).unwrap();
    }

    result
}

pub fn reduce_mean<A>(
    data: &ArrayBase<OwnedRepr<A>, IxDyn>,
    axes: Option<&[i64]>,
    keepdims: bool,
    noop_with_empty_axes: bool,
) -> ArrayBase<OwnedRepr<A>, IxDyn>
where
    A: Float + FromPrimitive,
{
    // Convert axes to usize, handling negative values
    let data_shape = data.shape();
    let rank = data_shape.len();
    let axes: Vec<usize> = match axes {
        Some(ax) => ax.iter()
            .map(|&a| if a < 0 { (rank as i64 + a) as usize } else { a as usize })
            .collect(),
        None => (0..rank).collect(),
    };

    // Se gli assi sono vuoti e noop_with_empty_axes è true, ritorna il tensore originale
    if axes.is_empty() && noop_with_empty_axes {
        return data.to_owned();
    }

    // Calcola la media lungo gli assi specificati
    let mut reduced = data.to_owned();
    for &axis in &axes {
        reduced = reduced.mean_axis(Axis(axis))
            .expect("Failed to compute mean along the axis");

        if !keepdims {
            reduced = reduced.index_axis_move(Axis(axis), 0);
        }
    }

    // Gestisce il mantenimento delle dimensioni ridotte
    if keepdims {
        let mut shape = data_shape.to_vec();
        for &axis in &axes {
            shape[axis] = 1;
        }
        reduced = reduced.into_shape(shape).unwrap();
    }

    reduced
}

// Funzione HardSwish
fn hardswish_element(x: f32) -> f32 {
    let alpha = 1.0 / 6.0;
    let beta = 0.5;
    x * f32::max(0.0, f32::min(1.0, alpha * x + beta))
}

// Applicazione di HardSwish a un tensore
pub fn hardswish<S>(x: &ArrayBase<S, IxDyn>) -> ArrayD<f32>
where
    S: Data<Elem = f32>,
{
    x.mapv(|elem| hardswish_element(elem))
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

pub fn concat<T>(tensors: Vec<&Array<T, IxDyn>>, axis: isize) -> Array<T, IxDyn>
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

fn calculate_padding(
    input_shape: &[i64],
    kernel_shape: &[i64],
    strides: &[i64],
    dilations: &[i64],
    auto_pad: Option<&str>,
    pads: Option<&[i64]>
) -> Vec<i64> {
    match auto_pad {
        Some(pad) if pad == "SAME_UPPER" || pad == "SAME_LOWER" => {
            let mut new_pads = vec![0; kernel_shape.len() * 2];
            for i in 0..kernel_shape.len() {
                let input_size = input_shape[i + 2];
                let filter_size = (kernel_shape[i] - 1) * dilations[i] + 1;
                let output_size = (input_size + strides[i] - 1) / strides[i];
                let total_padding = if output_size * strides[i] + filter_size > input_size + strides[i] {
                    output_size * strides[i] + filter_size - input_size - strides[i]
                } else {
                    0
                };
                if pad == "SAME_UPPER" {
                    new_pads[i * 2] = total_padding / 2;
                    new_pads[i * 2 + 1] = total_padding - new_pads[i * 2];
                } else { // SAME_LOWER
                    new_pads[i * 2 + 1] = total_padding / 2;
                    new_pads[i * 2] = total_padding - new_pads[i * 2 + 1];
                }
            }
            new_pads
        }
        Some(pad) if pad == "VALID" => {
            vec![0; kernel_shape.len() * 2]
        }
        _ => {
            pads.unwrap_or(&vec![0; input_shape.len() * 2]).to_vec()
        }
    }
}


fn apply_padding<T>(
    input: &Array<T, IxDyn>, 
    pads: &[i64]
) -> Array<T, IxDyn>
where
    T: Default + Clone,
{
    let mut padded_shape = input.raw_dim();
    
    for (i, &pad) in pads.iter().enumerate() {
        if i % 2 == 0 {  // Padding di inizio
            padded_shape[i / 2] += pad as usize;
        } else {  // Padding di fine
            padded_shape[i / 2] += pad as usize;
        }
    }

    let mut padded_input = Array::<T, _>::default(padded_shape);
    let slice_s = if pads.iter().all(|&pad| pad == 0) {
        s![.., .., .., ..] // No padding, use the entire dimension
    } else {
        // Calculate start and end indices for height and width dimensions
        let height_start = pads[0] as isize;
        let height_end = if pads[2] > 0 { -(pads[2] as isize) } else { isize::MAX };
        let width_start = pads[1] as isize;
        let width_end = if pads[3] > 0 { -(pads[3] as isize) } else { isize::MAX };
    
        s![.., .., height_start..height_end, width_start..width_end]
    };
    padded_input.slice_mut(slice_s).assign(input);

    padded_input
}

pub fn convolution<T>(
    input: &Array<T, IxDyn>,
    weights: &Array<T, IxDyn>,
    bias: Option<&Array<T, IxDyn>>,
    auto_pad: Option<&str>,
    dilations: &[i64],
    group: i64,
    kernel_shape: &[i64],
    pads: Option<&[i64]>,
    strides: &[i64],
) -> Array<T, IxDyn>
where
    T: Default + Clone + Add<Output = T> + Mul<Output = T> + Copy + Zero,
{
    let input_shape = input.dim();
    // println!("Input shape: {:?}", input_shape);

    let input_shape_vec: Vec<i64> = input_shape.slice().iter().map(|&dim| dim as i64).collect();
    let actual_pads = calculate_padding(&input_shape_vec, kernel_shape, strides, dilations, auto_pad, pads);
    // println!("Actual Pads: {:?}", actual_pads);

    let padded_input = apply_padding(input, &actual_pads);

    // Correct calculation of output dimensions
    let output_height = ((input_shape[2] as i64 - kernel_shape[0] + actual_pads[0] + actual_pads[2]) / strides[0] + 1) as usize;
    let output_width = ((input_shape[3] as i64 - kernel_shape[1] + actual_pads[1] + actual_pads[3]) / strides[1] + 1) as usize;


    // println!("Kernel shape: {:?}", kernel_shape);
    // println!("Strides: {:?}", strides);
    // println!("Dilations: {:?}", dilations);

    // println!("Output height: {}", output_height);
    // println!("Output width: {}", output_width);

    // Validate the output dimensions
    if output_height <= 0 || output_width <= 0 {
        panic!("Invalid output dimensions computed: height {}, width {}. Check kernel size, padding, strides, and input dimensions.", output_height, output_width);
    }   


    let output_channels = weights.dim()[0];
    let output_shape = vec![input_shape[0], output_channels, output_height, output_width];

    let mut output = Array::<T, _>::zeros(IxDyn(&output_shape));


    let num_groups = group as usize;
    let input_group_size = input_shape[1] / num_groups;
    let weight_group_size = weights.dim()[1] / num_groups;

    for n in 0..input_shape[0] {
        for g in 0..num_groups {
            let input_group_slice = s![.., g * input_group_size..(g + 1) * input_group_size, .., ..];
            let weight_group_slice = s![.., g * weight_group_size..(g + 1) * weight_group_size, .., ..];

            let input_group = padded_input.slice(input_group_slice);
            let weight_group = weights.slice(weight_group_slice);

            for out_y in 0..output_height {
                for out_x in 0..output_width {
                    let in_y_start = out_y * strides[0] as usize;
                    let in_x_start = out_x * strides[1] as usize;
            
                    for m in 0..weight_group_size {
                        let mut sum = T::zero();
                        for c in 0..input_group_size {
                            for ky in 0..kernel_shape[0] as usize {
                                for kx in 0..kernel_shape[1] as usize {
                                    let in_y = in_y_start + ky * dilations[0] as usize;
                                    let in_x = in_x_start + kx * dilations[1] as usize;
                        
                                    // Correcting the indices for input and weight tensors
                                    let input_idx = [n, g * input_group_size + c, in_y, in_x];
                                    let weight_idx = [m, c, ky, kx]; // Corrected weight index
                        
                                    sum = sum + input_group[input_idx] * weight_group[weight_idx];
                                }
                            }
                
                            if let Some(b) = bias {
                                sum = sum + b[[m + g * weight_group_size]];
                            }
                
                            output[[n, m + g * weight_group_size, out_y, out_x]] = sum;
                        }
                    }
                }
            }
        }
    }

    output
}


pub fn max_from_slice<T: PartialOrd + Copy> (slice: &ArrayViewD<T>) -> T {
    let mut max = slice.first().unwrap().clone();
    slice.for_each(|&x| if x>max {max=x} );
    max
}

pub fn max_pool<T: Sync + Send + Float> (tensor: &Array<T, IxDyn>, auto_pad: Option<&str>,
                                         ceil_mode: Option<bool>, dilations: Option<Vec<isize>>,
                                         mut kernel_shape: Vec<isize>, pads: Option<Vec<isize>>,
                                         storage_order: Option<bool>, strides: Option<Vec<isize>>) -> Result<Array<T, IxDyn>, Error> {
    //todo: auto_pad, ceil_mode, storage_order

    if kernel_shape.len() != tensor.ndim() { return Err(Error::KernelShapeError) }
    if kernel_shape.iter().any(|x| *x<=0) { return Err(Error::KernelShapeError) }

    let auto_pad = match auto_pad {
        None => AutoPad::NotSet,
        Some(s) => match s {
            "NOTESET" => AutoPad::NotSet,
            "SAME_UPPER" => AutoPad::SameUpper,
            "SAME_LOWER" => AutoPad::SameLower,
            "VALID" => AutoPad::Valid,
            _ => return Err(Error::AutoPadError)
        }
    };

    let ceil_mode = ceil_mode.unwrap_or(false);

    let dilations = dilations.unwrap_or(vec![1; tensor.ndim()]);
    if dilations.len() != tensor.ndim() { return Err(Error::DilationError) }
    if dilations.iter().any(|x| *x<=0) { return Err(Error::DilationError) }

    let pads = pads.unwrap_or(vec![0; 2*tensor.ndim()]);
    if pads.len() != 2*tensor.ndim() { return Err(Error::PadsError) }
    if pads.iter().any(|x| *x<0) { return Err(Error::PadsError) }

    let storage_order = storage_order.unwrap_or(false);

    let strides = strides.unwrap_or(vec![1; tensor.ndim()]);
    if strides.len() != tensor.ndim() { return Err(Error::StridesError) }
    if strides.iter().any(|x| *x<=0) { return Err(Error::StridesError) }

    //Padding
    let pads_begin = &pads[0..tensor.ndim()];
    let pads_end = &pads[tensor.ndim()..];
    let padded_tensor_shape = pads_begin.iter()
        .zip(tensor.shape()).zip(pads_end)
        .map(|((&a,&b),&c)| a as usize + b + c as usize)
        .collect::<Vec<_>>();
    let mut padded_tensor = ArrayD::<T>::zeros(IxDyn(&padded_tensor_shape));
    //Copy the input tensor in padded_tensor
    let paste_slice_info = pads_begin.iter().zip(tensor.shape())
        .map(|(&a, &b)| SliceInfoElem::Slice{start: a, step: 1, end: Some(a + b as isize)})
        .collect::<Vec<_>>();
    let mut paste_slice =
        padded_tensor.slice_mut::<SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn>>(SliceInfo::try_from(paste_slice_info).unwrap());
    paste_slice.iter_mut().zip(tensor).for_each(|(a, &b)| *a = b);

    //slice_shape[i] = tensor_shape[i] - dilation[i] * (kernel_shape[i] - 1)
    let slice_shape = padded_tensor.shape().iter()
        .zip(&kernel_shape).zip(&dilations)
        .map(|((&a, &b), &c)| a as isize - c*(b-1))
        .collect::<Vec<_>>();
    let slice_info = slice_shape.iter()
        .map(|&i| SliceInfoElem::Slice{start: 0, step: 1, end: Some(i)})
        .collect::<Vec<_>>();
    let slice =
        padded_tensor.slice::<SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn>>(SliceInfo::try_from(slice_info).unwrap());


    //Viene creato un tensore di slice, ogni slice contiene gli elementi che deve considerare l'operatore max
    //map index -> SliceInfo -> Slice
    let tensor_of_slices =
        slice.indexed_iter()
            .map( |(d, _)|
                d.as_array_view().iter()
                    .zip(&kernel_shape).zip(&dilations)
                    .map(|((&i, &j), &d)|
                        SliceInfoElem::Slice{start: i as isize, step: d, end: Some(i as isize + d*(j-1) + 1)})
                    .collect::<Vec<_>>()
            ).map(|info| padded_tensor.slice::<SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn>>(SliceInfo::try_from(info).unwrap()))
            .collect::<Vec< ArrayViewD<T>>>();

    //Conversione da vec ad ArrayD
    let tensor_of_slices = ArrayD::from_shape_vec(slice.shape(), tensor_of_slices).unwrap();

    //Viene creata una slice del tensore tenendo conto degli stride
    let strides_slice_info = strides.iter()
        .map(|&i| SliceInfoElem::Slice{start: 0, step: i as isize, end: None})
        .collect::<Vec<_>>();
    let strides_slice =
        tensor_of_slices.slice::<SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn>>(SliceInfo::try_from(strides_slice_info).unwrap());

    //Ad ogni slice viene applicato l'operatore max
    Ok(Zip::from(&strides_slice).par_map_collect(|x| max_from_slice(x)))
}


pub fn transpose<A>(tensor: &ArrayBase<OwnedRepr<A>, IxDyn>, axes: Option<Vec<usize>>) -> ArrayBase<OwnedRepr<A>, IxDyn>
where
    A: Clone,
{
    match axes {
        Some(axes) => tensor.to_owned().permuted_axes(axes),  // Clona il tensore qui
        None => tensor.t().to_owned(),
    }
}


pub fn split<A>(tensor: &ArrayBase<OwnedRepr<A>, IxDyn>, indices: Vec<usize>, axis: usize) -> Vec<ArrayBase<OwnedRepr<A>, IxDyn>>
where
    A: Clone,
{
    let mut sub_tensors = Vec::new();
    let mut start = 0;
    for &index in &indices {
        let end = index;
        let mut view = tensor.view();
        view.slice_axis_inplace(Axis(axis), (start..end).into());
        sub_tensors.push(view.to_owned());
        start = end;
    }

    // Aggiungi l'ultimo segmento
    let mut last_view = tensor.view();
    last_view.slice_axis_inplace(Axis(axis), (start..).into());
    sub_tensors.push(last_view.to_owned());

    sub_tensors
}

pub fn tile<A>(tensor: &ArrayBase<OwnedRepr<A>, IxDyn>, reps: &Vec<usize>) -> ArrayBase<OwnedRepr<A>, IxDyn>
where
    A: Clone,
{
    let mut tiled_tensor = tensor.clone();
    for (axis, &rep) in reps.iter().enumerate() {
        tiled_tensor = tiled_tensor.clone().into_shape(tiled_tensor.raw_dim().insert_axis(Axis(axis)) * rep).unwrap();
    }
    tiled_tensor
}

pub fn gather<A>(tensor: &ArrayBase<OwnedRepr<A>, IxDyn>, indices: &Vec<usize>, axis: usize) -> ArrayBase<OwnedRepr<A>, IxDyn>
where
    A: Clone,
{
    // Costruisci una lista di viste basate sugli indici
    let slices: Vec<_> = indices.iter()
                                .map(|&i| tensor.index_axis(Axis(axis), i))
                                .collect();
    
    // Usa la funzione `stack` per impilare le viste lungo l'asse specificato
    ndarray::stack(Axis(axis), &slices).unwrap()
}

pub fn gemm<S>(
    a: &ArrayBase<S, Ix2>,
    b: &ArrayBase<S, Ix2>,
    c: Option<&ArrayBase<S, Ix2>>,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool
) -> Array2<f32>
where
    S: Data<Elem = f32>,
{
    let a = if trans_a { a.t() } else { a.view() };
    let b = if trans_b { b.t() } else { b.view() };

    let mut result = a.dot(&b) * alpha;

    if let Some(c) = c {
        result = result + c * beta;
    }

    result
}

// Funzione Flatten
pub fn flatten<S>(input: &ArrayBase<S, IxDyn>, axis: isize) -> ArrayBase<OwnedRepr<S::Elem>, IxDyn>
where
    S: Data,
    S::Elem: Clone,
{
    let shape = input.shape();
    let rank = shape.len() as isize;

    // Gestione axis negativo
    let axis = if axis < 0 { rank + axis } else { axis };

    // Verifica che axis sia nel range valido
    if axis < 0 || axis > rank {
        panic!("Axis out of range");
    }

    // Calcola le nuove dimensioni
    let (dims_before_axis, dims_after_axis) = shape.split_at(axis as usize);
    let new_dim0 = dims_before_axis.iter().product::<usize>();
    let new_dim1 = dims_after_axis.iter().product::<usize>();

    // Reshape e conversione in array dinamico
    input.to_owned().into_shape((new_dim0, new_dim1)).unwrap().into_dyn()
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



#[cfg(test)]
mod tests {
    use super::*; // Importa tutto dal modulo genitore
    use ndarray::{Array, ArrayD, IxDyn}; // Import the necessary ndarray types

    #[test]
    fn test_convolution_basic() {
        // Ensure that `input` and `weights` are dynamic-dimension arrays
        let input_data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let input = Array::from_shape_vec((1, 1, 4, 4), input_data).unwrap().into_dyn();
        let weights = Array::from_shape_vec((1, 1, 3, 3), vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0]).unwrap().into_dyn();
        let bias = Some(Array::zeros(IxDyn(&[1])).into_dyn());

        let result = convolution(&input, &weights, bias.as_ref(), None, &[1, 1], 1, &[3, 3], None, &[1, 1]);

        // Ensure that `expected_output` is a dynamic-dimension array for comparison
        let expected_output: ArrayD<f32> = Array::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![5.0, 3.0, -1.0, -3.0]).unwrap().into_dyn();

// Now this assertion should work without type errors
        assert_eq!(result, expected_output);
    }
    
    // Other test functions...
}

