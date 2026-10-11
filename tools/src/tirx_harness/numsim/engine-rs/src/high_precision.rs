use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::runtime::{
    read_runtime_bytes, resolve_runtime_physical_access, write_runtime_bytes, RuntimeBuffer,
};
use crate::{
    EngineError, PhysicalAccessKind, PhysicalAccessSpace, PhysicalByteSpan, PhysicalMemory,
    RuntimeScalar, WarpContext,
};

pub trait Format: Copy + Send + Sync + 'static {
    const ID: u8;
    const WIDTH: usize;
    fn decode(bytes: &[u8]) -> f64;
    fn encode(value: f64, bytes: &mut [u8]);
}

macro_rules! float_format {
    ($name:ident, $id:expr, $width:expr, $bits:ty, $decode:expr, $encode:expr) => {
        #[derive(Clone, Copy, Debug)]
        pub struct $name;
        impl Format for $name {
            const ID: u8 = $id;
            const WIDTH: usize = $width;
            fn decode(bytes: &[u8]) -> f64 {
                let bits = <$bits>::from_le_bytes(bytes.try_into().expect("validated float width"));
                ($decode)(bits) as f64
            }
            fn encode(value: f64, bytes: &mut [u8]) {
                let bits: $bits = ($encode)(value);
                bytes.copy_from_slice(&bits.to_le_bytes());
            }
        }
    };
}

float_format!(F16, 1, 2, u16, crate::fp16_bits_to_f32, |value: f64| {
    crate::f32_to_fp16_bits(value as f32)
});
float_format!(Bf16, 2, 2, u16, crate::bf16_bits_to_f32, |value: f64| {
    crate::f32_to_bf16_bits(value as f32)
});
float_format!(F32, 3, 4, u32, f32::from_bits, |value: f64| (value as f32)
    .to_bits());
float_format!(F64, 4, 8, u64, f64::from_bits, f64::to_bits);
float_format!(
    E4m3,
    5,
    1,
    u8,
    crate::float8_e4m3fn_bits_to_f32,
    |value: f64| crate::f32_to_float8_e4m3fn_bits(value as f32)
);
float_format!(
    E8m0,
    6,
    1,
    u8,
    crate::float8_e8m0fnu_bits_to_f32,
    |value: f64| crate::f32_to_float8_e8m0fnu_bits(value as f32)
);

#[derive(Clone, Copy, Debug)]
pub struct Value<F: Format> {
    pub value: f64,
    format: PhantomData<F>,
}

impl<F: Format> Value<F> {
    pub fn new(value: f64) -> Self {
        Self {
            value,
            format: PhantomData,
        }
    }
}

impl<F: Format> RuntimeScalar for Value<F> {
    const BYTE_LEN: usize = F::WIDTH;
    const HIGH_FORMAT: u8 = F::ID;
    fn zero() -> Self {
        Self::new(0.0)
    }
    fn decode_le(bytes: &[u8]) -> Result<Self, EngineError> {
        if bytes.len() != F::WIDTH {
            return Err(EngineError::message("high precision float width mismatch"));
        }
        Ok(Self::new(F::decode(bytes)))
    }
    fn encode_le_into(self, target: &mut [u8]) {
        F::encode(self.value, target);
    }
    fn high_value(self) -> f64 {
        self.value
    }
    fn with_high_value(self, value: f64) -> Self {
        Self::new(value)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ShadowValue {
    pub(crate) width: usize,
    pub(crate) format: u8,
    pub(crate) value: f64,
}

type ShadowKey = (PhysicalAccessSpace, u64, usize);

#[derive(Default)]
pub(crate) struct ShadowMemory {
    enabled: AtomicBool,
    values: OnceLock<Mutex<BTreeMap<ShadowKey, ShadowValue>>>,
}

impl ShadowMemory {
    pub(crate) fn enable(&self) {
        self.enabled.store(true, Ordering::Relaxed);
    }
    pub(crate) fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }
    pub(crate) fn read(
        &self,
        space: PhysicalAccessSpace,
        span: PhysicalByteSpan,
        format: u8,
    ) -> Result<Option<f64>, EngineError> {
        let Some(values) = self.values.get() else {
            return Ok(None);
        };
        let values = values.lock().unwrap();
        Ok(Self::lookup(&values, space, span, Some(format))?.map(|value| value.value))
    }

    fn lookup(
        values: &BTreeMap<ShadowKey, ShadowValue>,
        space: PhysicalAccessSpace,
        span: PhysicalByteSpan,
        format: Option<u8>,
    ) -> Result<Option<ShadowValue>, EngineError> {
        let key = (space, span.allocation().get(), span.byte_offset());
        if let Some(value) = values.get(&key) {
            if value.width == span.byte_len() && format.is_none_or(|format| value.format == format)
            {
                return Ok(Some(*value));
            }
        }
        let first = (space, span.allocation().get(), 0);
        let last = (space, span.allocation().get(), span.byte_end());
        if values
            .range(first..last)
            .next_back()
            .is_some_and(|(key, value)| key.2 + value.width > span.byte_offset())
        {
            return Err(EngineError::message(format!(
                "high precision cannot interpret a partial or differently typed value at {space} {span}"
            )));
        }
        Ok(None)
    }

    pub(crate) fn copy_value(
        &self,
        space: PhysicalAccessSpace,
        span: PhysicalByteSpan,
    ) -> Result<Option<ShadowValue>, EngineError> {
        let Some(values) = self.values.get() else {
            return Ok(None);
        };
        Self::lookup(&values.lock().unwrap(), space, span, None)
    }

    pub(crate) fn replace_copy(
        &self,
        space: PhysicalAccessSpace,
        span: PhysicalByteSpan,
        value: Option<ShadowValue>,
    ) -> Result<(), EngineError> {
        if value.is_some_and(|value| value.width != span.byte_len()) {
            return Err(EngineError::message("high precision copy width mismatch"));
        }
        let values = if value.is_some() {
            self.values.get_or_init(Mutex::default)
        } else if let Some(values) = self.values.get() {
            values
        } else {
            return Ok(());
        };
        let mut values = values.lock().unwrap();
        Self::lookup(&values, space, span, None)?;
        let key = (space, span.allocation().get(), span.byte_offset());
        if let Some(value) = value {
            values.insert(key, value);
        } else {
            values.remove(&key);
        }
        Ok(())
    }

    pub(crate) fn write(
        &self,
        space: PhysicalAccessSpace,
        span: PhysicalByteSpan,
        format: u8,
        value: f64,
    ) -> Result<(), EngineError> {
        let mut values = self.values.get_or_init(Mutex::default).lock().unwrap();
        Self::lookup(&values, space, span, Some(format))?;
        values.insert(
            (space, span.allocation().get(), span.byte_offset()),
            ShadowValue {
                width: span.byte_len(),
                format,
                value,
            },
        );
        Ok(())
    }

    pub(crate) fn global_values(&self, allocation: u64) -> Vec<(usize, usize, u8, f64)> {
        let Some(values) = self.values.get() else {
            return Vec::new();
        };
        values
            .lock()
            .unwrap()
            .range(
                (PhysicalAccessSpace::Global, allocation, 0)
                    ..=(PhysicalAccessSpace::Global, allocation, usize::MAX),
            )
            .map(|(key, value)| (key.2, value.width, value.format, value.value))
            .collect()
    }
}

pub(crate) fn memory(physical: &PhysicalMemory, space: PhysicalAccessSpace) -> &ShadowMemory {
    if space == PhysicalAccessSpace::Global {
        &physical.global().high_precision
    } else {
        &physical.high_precision
    }
}

pub(crate) fn read<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    lane: usize,
    offset: usize,
) -> Result<T, EngineError> {
    let access = resolve_runtime_physical_access(
        context,
        buffer,
        lane,
        offset,
        T::BYTE_LEN,
        PhysicalAccessKind::Read,
    )?;
    let bytes = read_runtime_bytes(physical, context, buffer, lane, offset, T::BYTE_LEN)?;
    let native = T::decode_le(&bytes)?;
    decode(physical, access.space(), access.span(), native)
}

pub(crate) fn decode<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    space: PhysicalAccessSpace,
    span: PhysicalByteSpan,
    native: T,
) -> Result<T, EngineError> {
    let shadow = memory(physical, space);
    match shadow.read(space, span, T::HIGH_FORMAT)? {
        Some(value) if T::HIGH_FORMAT != 0 => Ok(native.with_high_value(value)),
        Some(_) => Ok(native),
        None => {
            shadow.write(
                space,
                span,
                T::HIGH_FORMAT,
                if T::HIGH_FORMAT != 0 {
                    native.high_value()
                } else {
                    0.0
                },
            )?;
            Ok(native)
        }
    }
}

pub(crate) fn write<T: RuntimeScalar>(
    physical: &PhysicalMemory,
    context: &WarpContext,
    buffer: &RuntimeBuffer,
    lane: usize,
    offset: usize,
    value: T,
) -> Result<(), EngineError> {
    let access = resolve_runtime_physical_access(
        context,
        buffer,
        lane,
        offset,
        T::BYTE_LEN,
        PhysicalAccessKind::Write,
    )?;
    memory(physical, access.space()).read(access.space(), access.span(), T::HIGH_FORMAT)?;
    write_runtime_bytes(physical, context, buffer, lane, offset, &value.encode_le())?;
    memory(physical, access.space()).write(
        access.space(),
        access.span(),
        T::HIGH_FORMAT,
        if T::HIGH_FORMAT != 0 {
            value.high_value()
        } else {
            0.0
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PhysicalAllocationId;

    fn span(offset: usize, width: usize) -> PhysicalByteSpan {
        PhysicalByteSpan::new(PhysicalAllocationId::new(1), offset, width).unwrap()
    }

    #[test]
    fn shadow_rejects_partial_and_reinterpreted_cells() {
        let shadow = ShadowMemory::default();
        let space = PhysicalAccessSpace::Global;
        shadow.write(space, span(2, 2), F16::ID, 1.00001).unwrap();
        assert_eq!(
            shadow.read(space, span(2, 2), F16::ID).unwrap(),
            Some(1.00001)
        );
        for access in [span(1, 2), span(2, 1), span(3, 2), span(0, 8)] {
            assert!(shadow.read(space, access, F16::ID).is_err());
            assert!(shadow.copy_value(space, access).is_err());
            assert!(shadow.write(space, access, F16::ID, 0.0).is_err());
        }
        assert!(shadow.read(space, span(2, 2), Bf16::ID).is_err());
        assert_eq!(shadow.read(space, span(4, 2), F16::ID).unwrap(), None);
        assert_eq!(
            shadow
                .read(PhysicalAccessSpace::Shared, span(2, 2), F16::ID)
                .unwrap(),
            None
        );
    }

    #[test]
    fn shadow_copy_preserves_values_and_clears_overwritten_cells() {
        let shadow = ShadowMemory::default();
        let space = PhysicalAccessSpace::Global;
        shadow.write(space, span(0, 2), F16::ID, 1.00001).unwrap();
        let captured = shadow.copy_value(space, span(0, 2)).unwrap();
        shadow.replace_copy(space, span(4, 2), captured).unwrap();
        assert_eq!(
            shadow.read(space, span(4, 2), F16::ID).unwrap(),
            Some(1.00001)
        );
        assert!(shadow.replace_copy(space, span(8, 4), captured).is_err());
        shadow.replace_copy(space, span(4, 2), None).unwrap();
        assert_eq!(shadow.read(space, span(4, 2), F16::ID).unwrap(), None);
        shadow.write(space, span(4, 2), Bf16::ID, 2.0).unwrap();
        assert_eq!(shadow.read(space, span(4, 2), Bf16::ID).unwrap(), Some(2.0));
    }

    #[test]
    fn concurrent_incompatible_writes_keep_shadow_cells_disjoint() {
        let shadow = ShadowMemory::default();
        let barrier = std::sync::Barrier::new(8);
        let successes = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|offset| {
                    let shadow = &shadow;
                    let barrier = &barrier;
                    scope.spawn(move || {
                        barrier.wait();
                        shadow.write(PhysicalAccessSpace::Global, span(offset, 8), F64::ID, 1.0)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap().is_ok())
                .filter(|success| *success)
                .count()
        });
        assert_eq!(successes, 1);
    }
}
