use std::path::Path;

use ndarray::{s, Array, Array4, Ix3};
use ort::{session::Session, value::Tensor};
use tokenizers::Tokenizer;

const VOCAB_SIZE: usize = 32064;

fn main() {
    let dylib_path = r"UPDATE WITH YOUR PATH \Local\Programs\Python\Python312\Lib\site-packages\onnxruntime\capi\onnxruntime.dll";
    ort::init_from(dylib_path).with_name("test").commit().unwrap();
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
	let tokenizer = Tokenizer::from_file(data_dir.join("tokenizer.json"))
        .unwrap();
	let session = Session::builder().unwrap()
		.with_execution_providers([
            // fail with DirectMLExecutionProvider but work with default
            
            // ort::execution_providers::DirectMLExecutionProvider::default()
            //     .build()
            //     .error_on_failure()
        ]).unwrap()
		.commit_from_file(data_dir.join("model.onnx"))
        .unwrap();

    let encoding = tokenizer.encode("hello world", true).unwrap();
    let token_ids: Vec<_> = encoding.get_ids().into_iter().map(|id| *id as i64).collect();
    let masks: Vec<_> = encoding.get_attention_mask().iter().map(|mask| *mask as i64).collect();
    let position_ids: Vec<_> = (0..token_ids.len() as i64).collect();

    let attention_mask = Array::from_shape_vec((1, masks.len()), masks).unwrap();
    let input_tensor = Array::from_shape_vec((1, token_ids.len()), token_ids).unwrap();
    let position_ids = Array::from_shape_vec((1, position_ids.len()), position_ids).unwrap();

    // Initialize past_key_values
    // This is used to store the attention mechanism's state across multiple inference steps
    // The structure is:
    // - 64 elements (32 layers, each with a key and value)
    // - Each element is a 4D array with dimensions:
    //   1. Batch size (1)
    //   2. Number of attention heads (32)
    //   3. Sequence length (0 initially, will grow with each token generated)
    //   4. Head size (96)
    let past_key_values: Vec<Array4<half::f16>> = vec![Array4::from_elem((1, 32, 0, 96), half::f16::ZERO); 64];

    let mut inputs = ort::inputs![
        "input_ids" => input_tensor.into_dyn().view(),
        "position_ids" => position_ids.into_dyn().view(),
        "attention_mask" => attention_mask.into_dyn().view(),
    ].unwrap();

    for i in 0..32 {
        let keys: Tensor<half::f16> = Tensor::from_array(past_key_values[i * 2].view()).unwrap();
        let values: Tensor<half::f16> = Tensor::from_array(past_key_values[i * 2 + 1].view()).unwrap();
        inputs.push((format!("past_key_values.{}.key", i).into(), keys.into()));
        inputs.push((format!("past_key_values.{}.value", i).into(), values.into()));
    }

    let outputs = session.run(inputs).expect("Expected success but failed");

    let logits: ndarray::ArrayView<_, _> = outputs["logits"].try_extract_tensor::<half::f16>().unwrap().into_dimensionality::<Ix3>().unwrap();
    let next_token_id = logits
        .slice(s![0, -1, ..VOCAB_SIZE])
        .into_iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0 as i64;
    println!("{next_token_id}");
}
