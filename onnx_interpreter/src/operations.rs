use core::panic;
use std::fmt::Display;
use ndarray::{ArrayD, Zip, Array, Array1, ArrayBase, IxDyn, Dimension, Axis, OwnedRepr, s, Data, ArrayViewD, SliceInfo, SliceInfoElem, Slice};
use std::ops::{Add, Div, Mul, Sub};
use num_traits::float::Float;
use std::cmp;
use std::convert::TryFrom;
use num_traits::{Zero, FromPrimitive};
use std::thread;
use std::sync::{Arc, Mutex, RwLock};

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

pub fn unsqueeze<A>(data: &Array<A, IxDyn>, axes: &Vec<i64>) -> ArrayD<A>
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

pub fn reshape<T: Clone>(data: &Array<T, IxDyn>, new_shape: &[i64]) -> Array<T, IxDyn> {
    let shape: Vec<usize> = new_shape.iter().enumerate().map(|(i, &dim)| {
        if dim == 0 {
            data.shape()[i]
        } else if dim == -1 {
            data.dim().size() / new_shape.iter().filter(|&&x| x != -1).product::<i64>() as usize
        } else {
            dim as usize
        }
    }).collect();

    Array::from_shape_vec(shape, data.iter().cloned().collect()).unwrap()
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
    _momentum: f32,
    training_mode: bool
) -> ArrayBase<OwnedRepr<f32>, IxDyn> {
    if training_mode {
        panic!("Training mode is not supported yet");
    }

    let mut output = ArrayBase::<OwnedRepr<f32>, IxDyn>::zeros(x.shape());

    for dim1 in 0..x.shape()[0] {
        for dim2 in 0..x.shape()[1] {
            for dim3 in 0..x.shape()[2] {
                for dim4 in 0..x.shape()[3] {
                    let x_value = x[[dim1, dim2, dim3, dim4]];
                    let scale_value = scale[[dim2]];
                    let b_value = b[[dim2]];
                    let input_mean_value = input_mean[[dim2]];
                    let input_var_value = input_var[[dim2]];

                    let sqrt_var = (input_var_value + epsilon).sqrt();
                    let norm = (x_value - input_mean_value) / sqrt_var;
                    let y = norm * scale_value + b_value;

                    output[[dim1, dim2, dim3, dim4]] = y;
                }
            }
        }
    }
    
    output
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

pub fn reduce_mean<T: Clone + Zero + FromPrimitive + Div<Output = T> + Display>(
    data: &Array<T, IxDyn>,
    axes: Option<Vec<i64>>,
    keepdims: bool,
    noop_with_empty_axes: bool,
) -> Array<T, IxDyn> {
    if axes.is_none() && noop_with_empty_axes {
        return data.clone();
    }

    let reduce_axes = axes.unwrap_or_else(|| (0..data.ndim() as i64).collect());
    let positive_axes: Vec<usize> = reduce_axes.iter()
        .map(|&axis| if axis < 0 { (data.ndim() as i64 + axis) as usize } else { axis as usize })
        .collect();

    println!("positive_axes: {:?}", positive_axes);

    let mut result = data.clone();
    for (index, &axis) in positive_axes.iter().enumerate() {
        if axis - index >= result.ndim() {
            panic!("Invalid axis: {}", axis);
        }
        result = match result.mean_axis(Axis(axis - index)) {
            Some(mean) => mean,
            None => panic!("Failed to calculate mean along axis {}", axis),
        };

        println!("result: {}", result);
    }

    if keepdims {
        let mut shape = data.shape().to_vec();
        for &axis in &positive_axes {
            shape[axis] = 1;
        }
        result = result.into_shape(IxDyn(&shape)).unwrap();
    }

    result
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

pub fn max_from_slice<T: PartialOrd + Copy> (slice: &ArrayViewD<T>) -> T {
    let mut max = slice.first().unwrap().clone();
    slice.for_each(|&x| if x>max {max=x} );
    max
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


pub fn split<T: Clone>(tensor: &Array<T, IxDyn>, indices: Vec<usize>, axis: usize) -> Vec<Array<T, IxDyn>> {
    if indices.iter().sum::<usize>() != tensor.shape()[axis] {
        panic!("La somma degli indici di split deve essere uguale alla dimensione dell'asse di split");
    }

    let mut split_tensors = Vec::new();
    let mut start = 0;
    for &index in &indices {
        let end = start + index;
        let split_tensor = tensor.slice_axis(Axis(axis), (start..end).into()).to_owned();
        split_tensors.push(split_tensor);
        start = end;
    }

    split_tensors
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
    a: &ArrayBase<S, IxDyn>,
    b: &ArrayBase<S, IxDyn>,
    c: Option<&ArrayBase<S, IxDyn>>,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool
) -> Result<Array<f32, IxDyn>, &'static str>
where
    S: Data<Elem = f32>,
{   
    let a_2 = a.view().into_dimensionality::<ndarray::Ix2>().map_err(|_| "Tensor A with wrong dimensionality")?;
    let b_2 = b.view().into_dimensionality::<ndarray::Ix2>().map_err(|_| "Tensor B with wrong dimensionality")?;  
    let a = if trans_a { a_2.t() } else { a_2.view() };
    let b = if trans_b { b_2.t() } else { b_2.view() };

    let mut result = a.dot(&b) * alpha;

    if let Some(c) = c {
        let c_2 = c.view().into_dimensionality::<ndarray::Ix1>().map_err(|_| "Tensor C with wrong dimensionality")?.to_owned();
        result = result + &c_2 * beta;
    }

    Ok(result.into_dyn())
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

pub fn convolution<T: 'static + Clone + Copy + Zero + Mul<Output = T> + Send + Sync>(
    input: &Array<T, IxDyn>,
    weights: &Array<T, IxDyn>,
    bias: Option<&Array<T, IxDyn>>,
    auto_pad: &str,
    dilations: &[i64],
    group: i64,
    kernel_shape: &[i64],
    pads: &[i64],
    strides: &[i64],
) -> Array<T, IxDyn> {
    if dilations != &[1, 1] {
        panic!("Dilation not supported yet");
    }

    if input.ndim() != 4 {
        panic!("Input tensor must have 4 dimensions (N x C x H x W)");
    }
    let real_pads = match auto_pad {
        "NOTSET" => pads.to_vec(),
        "SAME_UPPER" => {
            let mut new_pads = vec![0; kernel_shape.len() * 2];
            for i in 0..kernel_shape.len() {
                let input_size = input.shape()[i + 2] as i64;
                let filter_size = kernel_shape[i];
                let output_size = (input_size + strides[i] - 1) / strides[i];
                let total_padding = if output_size * strides[i] + filter_size > input_size + strides[i] {
                    output_size * strides[i] + filter_size - input_size - strides[i]
                } else {
                    0
                };
                new_pads[i] = total_padding / 2;
                new_pads[i + kernel_shape.len()] = total_padding - new_pads[i];
            }
            new_pads
        }
        "SAME_LOWER" => {
            let mut new_pads = vec![0; kernel_shape.len() * 2];
            for i in 0..kernel_shape.len() {
                let input_size = input.shape()[i + 2] as i64;
                let filter_size = kernel_shape[i];
                let output_size = (input_size + strides[i] - 1) / strides[i];
                let total_padding = if output_size * strides[i] + filter_size > input_size + strides[i] {
                    output_size * strides[i] + filter_size - input_size - strides[i]
                } else {
                    0
                };
                new_pads[i + kernel_shape.len()] = total_padding / 2;
                new_pads[i] = total_padding - new_pads[i + kernel_shape.len()];
            }
            new_pads
        }
        _ => panic!("Auto padding not supported yet"),
    };

    let out_dim = input.shape()[2..].iter()
        .enumerate()
        .map(|(i, &d) | (d - kernel_shape[i] as usize + real_pads[i] as usize + real_pads[i + input.ndim() - 2] as usize) / strides[i] as usize + 1)
        .collect::<Vec<_>>();
    let share_output = Arc::new(Mutex::new(Array::<T, IxDyn>::zeros(IxDyn(&[input.shape()[0], weights.shape()[0], out_dim[0], out_dim[1]]))));
    let share_input = Arc::new(RwLock::new(input.clone()));
    let mut threads = Vec::new();

    for n in 0..input.shape()[0] {
        for g in 0..group {
            for m in 0..weights.shape()[0] {
                let share_output_clone = Arc::clone(&share_output);
                let share_input_clone = Arc::clone(&share_input);
                let weights_clone = weights.clone();
                let bias_clone = match bias {
                    Some(b) => Some(b.clone()),
                    None => None,
                };
                let group_clone = group as usize;
                let kernel_shape_clone = kernel_shape.to_vec();
                let real_pads_clone = real_pads.to_vec();
                let strides_clone = strides.to_vec();
                let out_dim_clone = out_dim.to_vec();
                threads.push(thread::spawn(move || {
                    let input = share_input_clone.read().unwrap();
                    let mut output = share_output_clone.lock().unwrap();
                    for y in 0..out_dim_clone[0] {
                        for x in 0..out_dim_clone[1] {
                            let mut sum = T::zero();
                            for c in 0..input.shape()[1] / group_clone {
                                for ky in 0..kernel_shape_clone[0] {
                                    for kx in 0..kernel_shape_clone[1] {
                                        let in_y = y as i64 * strides_clone[0] + ky - real_pads_clone[0];
                                        let in_x = x as i64 * strides_clone[1] + kx - real_pads_clone[1];
                                        if in_y >= 0 && in_y < input.shape()[2] as i64 && in_x >= 0 && in_x < input.shape()[3] as i64 {
                                            let input_idx = [n, g as usize * input.shape()[1] / group_clone + c, in_y as usize, in_x as usize];
                                            let weight_idx = [m, c, ky as usize, kx as usize];
                                            sum = sum + input[input_idx] * weights_clone[weight_idx];
                                        }
                                    }
                                }
                            }
                            if let Some(b) = &bias_clone {
                                sum = sum + b[m];
                            }
                            output[[n, m, y, x]] = sum;
                        }
                    }
                }));

            }
        }
    }

    for thread in threads {
        thread.join().unwrap();
    }

    let output = share_output.lock().unwrap().clone();

    output
}

pub fn max_pool<T: 'static + Clone + Copy + Zero + PartialOrd + Send + Sync>(
    input: &Array<T, IxDyn>,
    auto_pad: &str,
    ceil_mode: bool,
    dilations: &[i64],
    kernel_shape: &[i64],
    pads: &[i64],
    storage_order: bool,
    strides: &[i64],
) -> Array<T, IxDyn> {
    if ceil_mode {
        panic!("Ceil mode not supported yet");
    }
    if dilations != &[1, 1] {
        panic!("Dilation not supported yet");
    }
    if storage_order {
        panic!("Storage order not supported yet");
    }
    if input.ndim() != 4 {
        panic!("Input tensor must have 4 dimensions (N x C x H x W)");
    }
    let real_pads = match auto_pad {
        "NOTSET" => pads.to_vec(),
        "SAME_UPPER" => {
            let mut new_pads = vec![0; kernel_shape.len() * 2];
            for i in 0..kernel_shape.len() {
                let input_size = input.shape()[i + 2] as i64;
                let filter_size = kernel_shape[i];
                let output_size = (input_size + strides[i] - 1) / strides[i];
                let total_padding = if output_size * strides[i] + filter_size > input_size + strides[i] {
                    output_size * strides[i] + filter_size - input_size - strides[i]
                } else {
                    0
                };
                new_pads[i] = total_padding / 2;
                new_pads[i + kernel_shape.len()] = total_padding - new_pads[i];
            }
            new_pads
        }
        "SAME_LOWER" => {
            let mut new_pads = vec![0; kernel_shape.len() * 2];
            for i in 0..kernel_shape.len() {
                let input_size = input.shape()[i + 2] as i64;
                let filter_size = kernel_shape[i];
                let output_size = (input_size + strides[i] - 1) / strides[i];
                let total_padding = if output_size * strides[i] + filter_size > input_size + strides[i] {
                    output_size * strides[i] + filter_size - input_size - strides[i]
                } else {
                    0
                };
                new_pads[i + kernel_shape.len()] = total_padding / 2;
                new_pads[i] = total_padding - new_pads[i + kernel_shape.len()];
            }
            new_pads
        }
        _ => panic!("Auto padding not supported yet"),
    };

    let out_dim = input.shape()[2..].iter()
        .enumerate()
        .map(|(i, &d) | (d - kernel_shape[i] as usize + real_pads[i] as usize + real_pads[i + input.ndim() - 2] as usize) / strides[i] as usize + 1)
        .collect::<Vec<_>>();
    let share_output = Arc::new(Mutex::new(Array::<T, IxDyn>::zeros(IxDyn(&[input.shape()[0], input.shape()[1], out_dim[0], out_dim[1]]))));
    let share_input = Arc::new(RwLock::new(input.clone()));
    let mut threads = Vec::new();

    for n in 0..input.shape()[0] {
        for c in 0..input.shape()[1] {
            let share_output_clone = Arc::clone(&share_output);
            let share_input_clone = Arc::clone(&share_input);
            let kernel_shape_clone = kernel_shape.to_vec();
            let real_pads_clone = real_pads.to_vec();
            let strides_clone = strides.to_vec();
            let out_dim_clone = out_dim.to_vec();
            threads.push(thread::spawn(move || {
                let input = share_input_clone.read().unwrap();
                let mut output = share_output_clone.lock().unwrap();
                for y in 0..out_dim_clone[0] {
                    for x in 0..out_dim_clone[1] {
                        let mut max = T::zero();
                        for ky in 0..kernel_shape_clone[0] {
                            for kx in 0..kernel_shape_clone[1] {
                                let in_y = y as i64 * strides_clone[0] + ky - real_pads_clone[0];
                                let in_x = x as i64 * strides_clone[1] + kx - real_pads_clone[1];
                                if in_y >= 0 && in_y < input.shape()[2] as i64 && in_x >= 0 && in_x < input.shape()[3] as i64 {
                                    let input_idx = [n, c, in_y as usize, in_x as usize];
                                    if input[input_idx] > max {
                                        max = input[input_idx];
                                    }
                                }
                            }
                        }
                        output[[n, c, y, x]] = max;
                    }
                }
            }));
        }
    }

    for thread in threads {
        thread.join().unwrap();
    }

    let output = share_output.lock().unwrap().clone();

    output
}
