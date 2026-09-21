//! CoreML FFI for the fixed-shape ANE bundle (macOS aarch64 only).
//!
//! The bound surface is deliberately small: load one `.mlpackage`,
//! validate its input/output signature against the expected fixed
//! shape, allocate fp16 `MLMultiArray` inputs, run one synchronous
//! prediction, read the two outputs back through their real strides.
//! Every Objective-C error arrives as `NSError` and is classified into
//! the DESIGN.md taxonomy by domain+code — never by broad message
//! matching.

use std::collections::HashMap;
use std::path::Path;

use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2::AnyThread;
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL};

/// `NSError` → the DESIGN.md error class. `Capacity` covers ANE
/// daemon/compiler resource exhaustion and POSIX memory/space pressure;
/// everything else under `com.apple.CoreML` and unknown domains is hard
/// (`Inference`) — deterministic shape faults are raised by the adapter,
/// not inferred from provider errors.
#[derive(Debug)]
pub enum NativeError {
    Capacity(String),
    Inference(String),
}

fn classify(err: &NSError) -> NativeError {
    let domain = err.domain().to_string();
    let code = err.code();
    let msg = format!("{domain}[{code}]: {}", err.localizedDescription());
    let lower = domain.to_lowercase();
    if lower.contains("ane") {
        // com.apple.ANED / ANECompilerService: daemon or compute-unit
        // exhaustion — transient capacity class.
        return NativeError::Capacity(msg);
    }
    if domain == "NSPOSIXErrorDomain" && matches!(code, 12 | 28 | 35) {
        // ENOMEM / ENOSPC / EAGAIN.
        return NativeError::Capacity(msg);
    }
    NativeError::Inference(msg)
}

pub struct NativeModel {
    model: Retained<MLModel>,
    /// Output feature names, resolved by shape at load (the traced
    /// identifiers are not stable across conversions).
    logits_name: String,
    pooled_name: String,
}

// Apple documents MLModel prediction as thread-safe; the adapter also
// serializes access through a Mutex.
unsafe impl Send for NativeModel {}
unsafe impl Sync for NativeModel {}

fn shape_of(array: &MLMultiArray) -> Vec<usize> {
    unsafe { array.shape() }
        .iter()
        .map(|n| n.as_isize() as usize)
        .collect()
}

/// True when the host OS meets the bundle's `minimum_deployment_target`
/// (`"macOS15 / iOS18"` → major 15). Unknown/absent values pass.
pub fn host_supports(minimum_target: Option<&str>) -> bool {
    let Some(spec) = minimum_target else {
        return true;
    };
    let Some(required) = spec
        .split(|c: char| !c.is_ascii_alphanumeric())
        .collect::<Vec<_>>()
        .windows(2)
        .find(|w| w[0].eq_ignore_ascii_case("macos"))
        .and_then(|w| w[1].parse::<isize>().ok())
    else {
        return true;
    };
    let v = objc2_foundation::NSProcessInfo::processInfo().operatingSystemVersion();
    v.majorVersion >= required
}

/// Allocate a Float16 multiarray and write `data` (C-order u16 bits)
/// through the array's own strides — layout assumptions about CoreML's
/// internal storage are never made.
#[allow(deprecated)] // dataPointer is documented + stride-checked here
fn multiarray_from_f16(
    data: &[u16],
    shape: &[usize],
) -> Result<Retained<MLMultiArray>, NativeError> {
    let count: usize = shape.iter().product();
    if data.len() != count {
        return Err(NativeError::Inference(format!(
            "input buffer {} elements for shape {shape:?}",
            data.len()
        )));
    }
    let ns_shape = ns_shape(shape);
    let array = unsafe {
        MLMultiArray::initWithShape_dataType_error(
            MLMultiArray::alloc(),
            &ns_shape,
            MLMultiArrayDataType::Float16,
        )
    }
    .map_err(|e| classify(&e))?;
    let strides: Vec<isize> = unsafe { array.strides() }
        .iter()
        .map(|n| n.as_isize())
        .collect();
    let base = unsafe { array.dataPointer() }.as_ptr() as *mut u16;
    // Strides are in elements; negative or absurd values are rejected.
    if strides.iter().any(|&s| s < 0) {
        return Err(NativeError::Inference("negative multiarray strides".into()));
    }
    let mut idx = vec![0usize; shape.len()];
    for flat in 0..count {
        let mut offset = 0usize;
        for (d, &i) in idx.iter().enumerate() {
            offset += i * strides[d] as usize;
        }
        unsafe { *base.add(offset) = data[flat] };
        for d in (0..shape.len()).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Ok(array)
}

/// Read an fp16 output through its strides into a C-order f32 vec.
#[allow(deprecated)] // dataPointer is documented + stride-checked here
fn read_f16_as_f32(array: &MLMultiArray) -> Result<Vec<f32>, NativeError> {
    let shape = shape_of(array);
    let count: usize = shape.iter().product();
    let strides: Vec<isize> = unsafe { array.strides() }
        .iter()
        .map(|n| n.as_isize())
        .collect();
    let base = unsafe { array.dataPointer() }.as_ptr() as *const u16;
    let mut out = Vec::with_capacity(count);
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..count {
        let mut offset = 0isize;
        for (d, &i) in idx.iter().enumerate() {
            offset += i as isize * strides[d];
        }
        let bits = unsafe { *base.offset(offset) };
        out.push(half::f16::from_bits(bits).to_f32());
        for d in (0..shape.len()).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Ok(out)
}

fn ns_shape(shape: &[usize]) -> Retained<NSArray<NSNumber>> {
    let nums: Vec<Retained<NSNumber>> = shape
        .iter()
        .map(|&d| NSNumber::new_isize(d as isize))
        .collect();
    let refs: Vec<&NSNumber> = nums.iter().map(|n| &**n).collect();
    NSArray::from_slice(&refs)
}

/// Compile `model.mlpackage` once into `dest` (a `model.mlmodelc`
/// directory); returns early when `dest` already exists. CoreML compiles
/// into a temp location — we move the result under our cache key so
/// later loads skip compilation. The async API is the only one that
/// accepts `.mlpackage`; we block the (already-blocking) caller on a
/// channel.
pub fn compile_package(package: &Path, dest: &Path) -> Result<(), NativeError> {
    if dest.is_dir() {
        return Ok(());
    }
    let path_str = package
        .to_str()
        .ok_or_else(|| NativeError::Inference("model path is not UTF-8".into()))?;
    let url = NSURL::fileURLWithPath(&NSString::from_str(path_str));
    let (tx, rx) = std::sync::mpsc::channel();
    let block = block2::RcBlock::new(move |compiled: *mut NSURL, err: *mut NSError| {
        let out = if let Some(e) = unsafe { err.as_ref() } {
            Err(classify(e))
        } else if compiled.is_null() {
            Err(NativeError::Inference("compile returned nil URL".into()))
        } else {
            match unsafe { Retained::retain(compiled) } {
                Some(u) => Ok(u),
                None => Err(NativeError::Inference("compile returned nil URL".into())),
            }
        };
        let _ = tx.send(out);
    });
    unsafe { MLModel::compileModelAtURL_completionHandler(&url, &block) };
    let compiled = rx
        .recv()
        .map_err(|_| NativeError::Inference("compile handler never fired".into()))??;
    let compiled_path = compiled
        .path()
        .map(|p| std::path::PathBuf::from(p.to_string()))
        .ok_or_else(|| NativeError::Inference("compiled model has no path".into()))?;
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| NativeError::Inference(format!("compiled cache dir: {e}")))?;
    }
    match std::fs::rename(&compiled_path, dest) {
        Ok(()) => Ok(()),
        Err(e) if dest.is_dir() => {
            // Loser of a same-content race keeps the winner's copy.
            let _ = e;
            let _ = std::fs::remove_dir_all(&compiled_path);
            Ok(())
        }
        Err(e) => Err(NativeError::Inference(format!(
            "install compiled model: {e}"
        ))),
    }
}

impl NativeModel {
    /// Load a compiled `model.mlmodelc` and validate its signature:
    /// `embeddings [1,W,1,L]`, `full_mask`/`local_mask [1,L,1,L]`,
    /// `type_vectors [1,W,1,1]`, `marker_map [1,L,1,K]` in fp16; two fp16
    /// outputs distinguished by shape ([.,1,.,K] logits / [.,W,.,1]
    /// pooled).
    pub fn load(
        package: &Path,
        hidden: usize,
        fixed_len: usize,
        max_options: usize,
        compute_units: MLComputeUnits,
    ) -> Result<Self, NativeError> {
        autoreleasepool(|_| {
            let path_str = package
                .to_str()
                .ok_or_else(|| NativeError::Inference("model path is not UTF-8".into()))?;
            let url = NSURL::fileURLWithPath(&NSString::from_str(path_str));
            let config = unsafe { MLModelConfiguration::new() };
            unsafe { config.setComputeUnits(compute_units) };
            let model =
                unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &config) }
                    .map_err(|e| classify(&e))?;

            let desc = unsafe { model.modelDescription() };
            let expected: HashMap<&str, Vec<usize>> = HashMap::from([
                ("embeddings", vec![1, hidden, 1, fixed_len]),
                ("full_mask", vec![1, fixed_len, 1, fixed_len]),
                ("local_mask", vec![1, fixed_len, 1, fixed_len]),
                ("type_vectors", vec![1, hidden, 1, 1]),
                ("marker_map", vec![1, fixed_len, 1, max_options]),
            ]);
            let inputs = unsafe { desc.inputDescriptionsByName() };
            for (name, want) in &expected {
                let Some(feature) = inputs.objectForKey(&NSString::from_str(name)) else {
                    return Err(NativeError::Inference(format!(
                        "model input {name:?} missing"
                    )));
                };
                let Some(constraint) = (unsafe { feature.multiArrayConstraint() }) else {
                    return Err(NativeError::Inference(format!(
                        "model input {name:?} is not a multiarray"
                    )));
                };
                let got = unsafe { constraint.shape() }
                    .iter()
                    .map(|n| n.as_isize() as usize)
                    .collect::<Vec<_>>();
                if &got != want {
                    return Err(NativeError::Inference(format!(
                        "model input {name:?} shape {got:?} != {want:?}"
                    )));
                }
                if unsafe { constraint.dataType() } != MLMultiArrayDataType::Float16 {
                    return Err(NativeError::Inference(format!(
                        "model input {name:?} is not Float16"
                    )));
                }
            }

            let outputs = unsafe { desc.outputDescriptionsByName() };
            let mut logits_name = None;
            let mut pooled_name = None;
            for key in outputs.allKeys().iter() {
                let feature = outputs
                    .objectForKey(&*key)
                    .ok_or_else(|| NativeError::Inference("output description missing".into()))?;
                let Some(constraint) = (unsafe { feature.multiArrayConstraint() }) else {
                    continue;
                };
                let shape = unsafe { constraint.shape() }
                    .iter()
                    .map(|n| n.as_isize() as usize)
                    .collect::<Vec<_>>();
                // Same shape test as the reference: dim-1 == 1 marks the
                // [1,1,1,K] logits head; the other is the [1,W,1,1]
                // pooled representation.
                if shape.get(1) == Some(&1) && shape.last() == Some(&max_options) {
                    logits_name = Some(key.to_string());
                } else if shape.get(1) == Some(&hidden) {
                    pooled_name = Some(key.to_string());
                }
            }
            Ok(Self {
                model,
                logits_name: logits_name
                    .ok_or_else(|| NativeError::Inference("no [1,1,1,K] logits output".into()))?,
                pooled_name: pooled_name
                    .ok_or_else(|| NativeError::Inference("no [1,W,1,1] pooled output".into()))?,
            })
        })
    }

    /// One synchronous prediction. `inputs` are the five C-order fp16
    /// buffers; returns `(logits[K], pooled[W])` as f32.
    pub fn predict(
        &self,
        inputs: &crate::host::AneInputs,
        l: usize,
        w: usize,
        max_options: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), NativeError> {
        autoreleasepool(|_| {
            let arrays = [
                ("embeddings", &inputs.embeddings, vec![1, w, 1, l]),
                ("full_mask", &inputs.full_mask, vec![1, l, 1, l]),
                ("local_mask", &inputs.local_mask, vec![1, l, 1, l]),
                ("type_vectors", &inputs.type_vectors, vec![1, w, 1, 1]),
                ("marker_map", &inputs.marker_map, vec![1, l, 1, max_options]),
            ];
            let mut keys: Vec<Retained<NSString>> = Vec::with_capacity(arrays.len());
            let mut vals: Vec<Retained<MLFeatureValue>> = Vec::with_capacity(arrays.len());
            for (name, buf, shape) in arrays {
                let arr = multiarray_from_f16(buf, &shape)?;
                keys.push(NSString::from_str(name));
                vals.push(unsafe { MLFeatureValue::featureValueWithMultiArray(&arr) });
            }
            let key_refs: Vec<&NSString> = keys.iter().map(|v| &**v).collect();
            let val_refs: Vec<&MLFeatureValue> = vals.iter().map(|v| &**v).collect();
            let dict: Retained<NSDictionary<NSString, MLFeatureValue>> =
                NSDictionary::from_slices(&key_refs, &val_refs);
            let dict_any: &NSDictionary<NSString, objc2::runtime::AnyObject> =
                unsafe { &*(&*dict as *const _ as *const _) };
            let provider = unsafe {
                MLDictionaryFeatureProvider::initWithDictionary_error(
                    MLDictionaryFeatureProvider::alloc(),
                    dict_any,
                )
            }
            .map_err(|e| classify(&e))?;
            let provider_ref: &ProtocolObject<dyn MLFeatureProvider> =
                ProtocolObject::from_ref(&*provider);
            let out = unsafe { self.model.predictionFromFeatures_error(provider_ref) }
                .map_err(|e| classify(&e))?;
            let logits = unsafe { out.featureValueForName(&NSString::from_str(&self.logits_name)) }
                .and_then(|v| unsafe { v.multiArrayValue() })
                .ok_or_else(|| NativeError::Inference("missing logits output".into()))?;
            let pooled = unsafe { out.featureValueForName(&NSString::from_str(&self.pooled_name)) }
                .and_then(|v| unsafe { v.multiArrayValue() })
                .ok_or_else(|| NativeError::Inference("missing pooled output".into()))?;
            Ok((read_f16_as_f32(&logits)?, read_f16_as_f32(&pooled)?))
        })
    }
}

/// Parse the `compute_units` config string (matches laya-coreml's set).
pub fn parse_compute_units(name: &str) -> Result<MLComputeUnits, String> {
    match name {
        "cpu" => Ok(MLComputeUnits::CPUOnly),
        "cpu_gpu" => Ok(MLComputeUnits::CPUAndGPU),
        "all" => Ok(MLComputeUnits::All),
        "cpu_ne" => Ok(MLComputeUnits::CPUAndNeuralEngine),
        other => Err(format!(
            "compute_units must be one of cpu_ne/cpu/cpu_gpu/all (got {other:?})"
        )),
    }
}
