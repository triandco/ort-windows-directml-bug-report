use std::path::Path;

use ndarray::{s, Array, Array4, Ix3};
use ort::{session::Session, value::Tensor};
use tokenizers::Tokenizer;

const VOCAB_SIZE: usize = 32064;

/// A simple program to run an inference session with the string "hello world".
/// ## Prerequisite
/// 1. Manually download Phi-3-mini-4k-instruct-onnx for directml model or use the ```data/download.sh```
/// 1. Install onnx runtime for directml 
/// using the command: ```pip install onnxruntime-directml```
/// this shall install the latest: onnxruntime-directml-1.20.1
/// 1. update dylib_path below
fn main() {
    let dylib_path = r"UPDATE WITH YOUR PATH \Local\Programs\Python\Python312\Lib\site-packages\onnxruntime\capi\onnxruntime.dll";
    ort::init_from(dylib_path).with_name("test").commit().unwrap();

    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
	let tokenizer = Tokenizer::from_file(data_dir.join("tokenizer.json"))
        .unwrap();
	let session = Session::builder().unwrap()
		.with_execution_providers([
            // works with default but fail with DirectMLExecutionProvider
            ort::execution_providers::DirectMLExecutionProvider::default()
                .build()
                .error_on_failure()
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
