use std::{fmt, sync::Arc};

use crate::physical_access::ProxyMemoryDomain;
use crate::{
    f32_to_tf32, AllocationId, BufferView, DeferredGlobalReduction, DeferredGlobalWrite,
    EngineError, GlobalMemory, MemoryAccessSemantics, OperationContext, PhysicalAccessBatch,
    PhysicalAccessKind, PhysicalBarrierId, PhysicalByteSpan, PhysicalMemory, WarpContext, WarpMask,
};

use super::{
    read_runtime_bytes, resolve_runtime_physical_access,
    resolve_shared_runtime_physical_access_to_cta, single_lane_physical_access_batch,
    single_lane_physical_access_batch_unmerged, single_lane_transfer_run_batch,
    PhysicalMbarrierCompletionTargets, PhysicalPtr, RuntimeBuffer,
};

#[derive(Clone)]
pub struct RuntimeTensorMap {
    view: BufferView,
    global_shape: Vec<usize>,
    global_strides: Vec<usize>,
    physical_global_shape: [usize; 5],
    physical_global_strides: [usize; 4],
    box_shape: Vec<usize>,
    element_strides: Vec<usize>,
    traversal_shape: Vec<usize>,
    element_bits: usize,
    interleave_bytes: Option<usize>,
    element_type: TensorMapElementType,
    fp4_shared_layout: Option<Fp4SharedLayout>,
    swizzle_bytes: Option<usize>,
    swizzle_atomicity: SwizzleAtomicity,
    fill_mode: TensorMapFillMode,
    transfer_template: Arc<TensorMapTransferTemplate>,
    im2col: Option<TensorMapIm2col>,
}

/// Im2col adds spatial bounds to a two-dimensional (channels, pixels) box.
/// Channels/pixels remain owned by box_shape, not duplicated in this metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
struct TensorMapIm2col {
    lower: [i16; 3],
    upper: [i16; 3],
    wide: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Im2colMode {
    Spatial,
    Wide,
    Wide128,
}

pub(super) const TENSOR_MAP_DESCRIPTOR_BYTES: usize = 128;
pub(super) const TENSOR_MAP_PAYLOAD_BYTES: usize = 64;
pub(super) const TENSOR_MAP_MAGIC: u8 = 0xA7;
pub(super) fn tensor_map_has_magic(byte: u8) -> bool {
    matches!(byte & !0x58, 0xA5..=0xA7)
}
fn tensor_map_payload_bytes(tag: u8) -> usize {
    if tag & 0x10 != 0 {
        80
    } else {
        TENSOR_MAP_PAYLOAD_BYTES
    }
}
const TENSOR_MAP_FLAG_MAGIC: u8 = 0x80;
const MAX_GLOBAL_DIMENSION: u128 = 1_u128 << 32;
const MAX_GLOBAL_STRIDE: u128 = 1_u128 << 40;
const MAX_BOX_DIMENSION: usize = 256;
const MAX_ELEMENT_STRIDE: usize = 8;

/// Independent of swizzle width so replace.swizzle_mode preserves this field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SwizzleAtomicity {
    B16,
    B32,
    B32Flip8,
    B64,
}

impl SwizzleAtomicity {
    fn bytes(self) -> usize {
        match self {
            Self::B16 => 16,
            Self::B32 | Self::B32Flip8 => 32,
            Self::B64 => 64,
        }
    }

    fn tag_bits(self) -> u8 {
        match self {
            Self::B16 => 0,
            Self::B32 => 8,
            Self::B32Flip8 => 64,
            Self::B64 => 72,
        }
    }

    fn from_tag(tag: u8) -> Self {
        match tag & 0x48 {
            0 => Self::B16,
            8 => Self::B32,
            64 => Self::B32Flip8,
            _ => Self::B64,
        }
    }
}

/// NumSim's private, byte-addressable TensorMap representation.
///
/// NVIDIA deliberately leaves the hardware CUtensorMap layout opaque. NumSim
/// keeps the architectural 128-byte object and ordering footprint, while its
/// supported tiled TensorMap semantics are self-contained in the first 64
/// bytes. The remaining bytes are reserved and ignored by direct decoding,
/// while automatic descriptor discovery requires their canonical zero value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct RuntimeTensorMapImage {
    allocation_id: u64,
    base_byte_offset: usize,
    host_address: bool,
    rank: usize,
    physical_global_shape: [usize; 5],
    physical_global_strides: [usize; 4],
    box_shape: [usize; 5],
    element_strides: [usize; 5],
    element_type: TensorMapElementType,
    interleave_bytes: Option<usize>,
    fp4_shared_layout: Option<Fp4SharedLayout>,
    swizzle_bytes: Option<usize>,
    swizzle_atomicity: SwizzleAtomicity,
    fill_mode: TensorMapFillMode,
    im2col: Option<TensorMapIm2col>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp4SharedLayout {
    Align8Packed,
    Align16Padded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorMapFillMode {
    Zero,
    OobNan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorMapElementType {
    Float4E2M1Fn,
    U6,
    Bool,
    I8,
    U8,
    Float8E4M3Fn,
    Float8E8M0Fnu,
    I16,
    U16,
    F16,
    Bf16,
    I32,
    U32,
    F32,
    Tf32,
    F64,
    I64,
    U64,
    U32x2,
    F32Ftz,
    Tf32Ftz,
}

impl TensorMapElementType {
    pub(crate) const fn bits(self) -> usize {
        match self {
            Self::Float4E2M1Fn => 4,
            Self::U6 => 6,
            Self::Bool | Self::I8 | Self::U8 | Self::Float8E4M3Fn | Self::Float8E8M0Fnu => 8,
            Self::I16 | Self::U16 | Self::F16 | Self::Bf16 => 16,
            Self::I32 | Self::U32 | Self::F32 | Self::Tf32 | Self::F32Ftz | Self::Tf32Ftz => 32,
            Self::F64 | Self::I64 | Self::U64 | Self::U32x2 => 64,
        }
    }

    const fn supports_oob_nan(self) -> bool {
        matches!(
            self,
            Self::F16
                | Self::Bf16
                | Self::F32
                | Self::Tf32
                | Self::F64
                | Self::F32Ftz
                | Self::Tf32Ftz
        )
    }

    const fn image_code(self) -> u8 {
        match self {
            Self::Float4E2M1Fn => 0,
            Self::Bool => 1,
            Self::I8 => 2,
            Self::U8 => 3,
            Self::Float8E4M3Fn => 4,
            Self::Float8E8M0Fnu => 5,
            Self::I16 => 6,
            Self::U16 => 7,
            Self::F16 => 8,
            Self::Bf16 => 9,
            Self::I32 => 10,
            Self::U32 => 11,
            Self::F32 => 12,
            Self::Tf32 => 13,
            Self::F64 => 14,
            Self::I64 => 15,
            Self::U64 => 16,
            Self::U32x2 => 17,
            Self::F32Ftz => 18,
            Self::Tf32Ftz => 19,
            Self::U6 => 20,
        }
    }

    fn from_image_code(code: u8) -> Result<Self, EngineError> {
        Ok(match code {
            0 => Self::Float4E2M1Fn,
            1 => Self::Bool,
            2 => Self::I8,
            3 => Self::U8,
            4 => Self::Float8E4M3Fn,
            5 => Self::Float8E8M0Fnu,
            6 => Self::I16,
            7 => Self::U16,
            8 => Self::F16,
            9 => Self::Bf16,
            10 => Self::I32,
            11 => Self::U32,
            12 => Self::F32,
            13 => Self::Tf32,
            14 => Self::F64,
            15 => Self::I64,
            16 => Self::U64,
            17 => Self::U32x2,
            18 => Self::F32Ftz,
            19 => Self::Tf32Ftz,
            20 => Self::U6,
            _ => {
                return Err(EngineError::message(format!(
                    "NumSim TensorMap image has unknown element-type code {code}"
                )));
            }
        })
    }
}

impl fmt::Display for TensorMapElementType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Float4E2M1Fn => "float4_e2m1fn",
            Self::U6 => "uint6",
            Self::Bool => "bool",
            Self::I8 => "int8",
            Self::U8 => "uint8",
            Self::Float8E4M3Fn => "float8_e4m3fn",
            Self::Float8E8M0Fnu => "float8_e8m0fnu",
            Self::I16 => "int16",
            Self::U16 => "uint16",
            Self::F16 => "float16",
            Self::Bf16 => "bfloat16",
            Self::I32 => "int32",
            Self::U32 => "uint32",
            Self::F32 => "float32",
            Self::Tf32 => "tf32",
            Self::F64 => "float64",
            Self::I64 => "int64",
            Self::U64 => "uint64",
            Self::U32x2 => "uint32x2",
            Self::F32Ftz => "float32_ftz",
            Self::Tf32Ftz => "tf32_ftz",
        })
    }
}

impl RuntimeTensorMapImage {
    pub(super) fn from_tensor_map(tensor_map: &RuntimeTensorMap) -> Self {
        let mut box_shape = [1_usize; 5];
        box_shape[..tensor_map.box_shape.len()].copy_from_slice(&tensor_map.box_shape);
        let mut element_strides = [1_usize; 5];
        element_strides[..tensor_map.rank()].copy_from_slice(&tensor_map.element_strides);
        Self {
            allocation_id: tensor_map.view.allocation().as_u64(),
            base_byte_offset: tensor_map.view.byte_offset(),
            host_address: false,
            rank: tensor_map.rank(),
            physical_global_shape: tensor_map.physical_global_shape,
            physical_global_strides: tensor_map.physical_global_strides,
            box_shape,
            element_strides,
            element_type: tensor_map.element_type,
            interleave_bytes: tensor_map.interleave_bytes,
            fp4_shared_layout: tensor_map.fp4_shared_layout,
            swizzle_bytes: tensor_map.swizzle_bytes,
            swizzle_atomicity: tensor_map.swizzle_atomicity,
            fill_mode: tensor_map.fill_mode,
            im2col: tensor_map.im2col.clone(),
        }
    }

    pub(super) fn encode(&self) -> Result<Vec<u8>, EngineError> {
        if self.rank == 0 || self.rank > 5 {
            return Err(EngineError::message(format!(
                "NumSim TensorMap image rank must be in 1..=5, got {}",
                self.rank
            )));
        }
        let base_byte_offset = u64::try_from(self.base_byte_offset)
            .map_err(|_| EngineError::message("TensorMap base byte offset does not fit u64"))?;
        let mut bytes = vec![
            0_u8;
            if self.im2col.is_some() {
                80
            } else {
                TENSOR_MAP_PAYLOAD_BYTES
            }
        ];
        bytes[0..8].copy_from_slice(&self.allocation_id.to_le_bytes());
        bytes[8..16].copy_from_slice(&base_byte_offset.to_le_bytes());

        for (axis, dimension) in self.physical_global_shape.iter().copied().enumerate() {
            if dimension == 0 || dimension as u128 > MAX_GLOBAL_DIMENSION {
                return Err(EngineError::message(format!(
                    "TensorMap global dimension {axis}={dimension} is outside 1..=2^32"
                )));
            }
            let encoded = if dimension as u128 == MAX_GLOBAL_DIMENSION {
                0_u32
            } else {
                u32::try_from(dimension).expect("bounded TensorMap dimension fits u32")
            };
            let start = 16 + axis * 4;
            bytes[start..start + 4].copy_from_slice(&encoded.to_le_bytes());
        }

        let mut encoded_strides = [0_u64; 4];
        for (axis, stride) in self.physical_global_strides.iter().copied().enumerate() {
            if stride as u128 >= MAX_GLOBAL_STRIDE || (stride != 0 && stride % 16 != 0) {
                return Err(EngineError::message(format!(
                    "TensorMap global stride {axis}={stride} is not zero or a 16-byte multiple below 2^40"
                )));
            }
            encoded_strides[axis] =
                u64::try_from(stride as u128 >> 4).expect("bounded TensorMap stride units fit u64");
        }
        for pair in 0..2 {
            let packed = u128::from(encoded_strides[pair * 2])
                | (u128::from(encoded_strides[pair * 2 + 1]) << 36);
            let start = 36 + pair * 9;
            bytes[start..start + 9].copy_from_slice(&packed.to_le_bytes()[..9]);
        }

        for (axis, dimension) in self.box_shape.iter().copied().enumerate() {
            let limit = if self.im2col.is_some() && axis == 1 {
                1024
            } else {
                MAX_BOX_DIMENSION
            };
            if dimension == 0 || dimension > limit {
                return Err(EngineError::message(format!(
                    "TensorMap box dimension {axis}={dimension} is outside 1..={limit}"
                )));
            }
            bytes[54 + axis] = ((dimension - 1) & 255) as u8;
        }

        bytes[59] = u8::try_from(self.rank).expect("bounded TensorMap rank fits u8")
            | (self.element_type.image_code() << 3);
        let fp4 = match self.fp4_shared_layout {
            None => 0_u8,
            Some(Fp4SharedLayout::Align8Packed) => 1,
            Some(Fp4SharedLayout::Align16Padded) => 2,
        };
        let swizzle = match self.swizzle_bytes {
            None => 0_u8,
            Some(32) => 1,
            Some(64) => 2,
            Some(128) => 3,
            Some(96) => 4,
            Some(value) => {
                return Err(EngineError::message(format!(
                    "NumSim TensorMap image cannot encode {value}B swizzle"
                )));
            }
        };
        bytes[60] = fp4
            | ((swizzle & 3) << 2)
            | ((swizzle & 4) << 4)
            | (u8::from(self.fill_mode == TensorMapFillMode::OobNan) << 4)
            | (u8::from(self.host_address) << 5)
            | TENSOR_MAP_FLAG_MAGIC;

        let mut encoded_element_strides = 0_u16;
        for (axis, stride) in self.element_strides.iter().copied().enumerate() {
            if stride == 0 || stride > MAX_ELEMENT_STRIDE {
                return Err(EngineError::message(format!(
                    "TensorMap element stride {axis}={stride} is outside 1..=8"
                )));
            }
            encoded_element_strides |= u16::try_from(stride - 1)
                .expect("bounded TensorMap element stride minus one fits u16")
                << (axis * 3);
        }
        bytes[61..63].copy_from_slice(&encoded_element_strides.to_le_bytes());
        // The format tag retains A7 for ordinary tiled maps; A6/A5 encode
        // 16B/32B interleave without competing metadata in the reserved tail.
        bytes[63] = match self.interleave_bytes {
            None => TENSOR_MAP_MAGIC,
            Some(16) => 0xA6,
            Some(32) => 0xA5,
            _ => {
                return Err(EngineError::message(
                    "TensorMap interleave must be 16B or 32B",
                ))
            }
        };
        bytes[63] |= self.swizzle_atomicity.tag_bits();
        if let Some(im2col) = &self.im2col {
            bytes[63] |= 0x10;
            for (i, value) in im2col.lower.iter().chain(&im2col.upper).enumerate() {
                bytes[64 + i * 2..66 + i * 2].copy_from_slice(&value.to_le_bytes());
            }
            bytes[76] = ((self.box_shape[1] - 1) >> 8) as u8 | (u8::from(im2col.wide) << 2);
        }
        Ok(bytes)
    }

    pub(super) fn decode(bytes: &[u8]) -> Result<Self, EngineError> {
        if bytes.len() < TENSOR_MAP_PAYLOAD_BYTES
            || bytes.len() != tensor_map_payload_bytes(bytes[63])
        {
            return Err(EngineError::message(format!(
                "NumSim TensorMap payload length disagrees with format tag: {} bytes",
                bytes.len()
            )));
        }
        if !tensor_map_has_magic(bytes[63]) {
            return Err(EngineError::message(
                "NumSim TensorMap image has invalid magic",
            ));
        }
        let rank = usize::from(bytes[59] & 0b111);
        if rank == 0 || rank > 5 {
            return Err(EngineError::message(format!(
                "NumSim TensorMap image rank must be in 1..=5, got {rank}"
            )));
        }
        let element_type = TensorMapElementType::from_image_code(bytes[59] >> 3)?;
        let interleave_bytes = match bytes[63] & !0x58 {
            0xA6 => Some(16),
            0xA5 => Some(32),
            _ => None,
        };
        let flags = bytes[60];
        if flags & 0x80 != TENSOR_MAP_FLAG_MAGIC {
            return Err(EngineError::message(
                "NumSim TensorMap image has invalid flag magic",
            ));
        }
        let fp4_shared_layout = match flags & 0b11 {
            0 => None,
            1 => Some(Fp4SharedLayout::Align8Packed),
            2 => Some(Fp4SharedLayout::Align16Padded),
            value => {
                return Err(EngineError::message(format!(
                    "NumSim TensorMap image has unknown FP4 layout code {value}"
                )));
            }
        };
        let swizzle_bytes = match ((flags >> 2) & 0b11) | ((flags >> 4) & 4) {
            0 => None,
            1 => Some(32),
            2 => Some(64),
            3 => Some(128),
            4 => Some(96),
            _ => {
                return Err(EngineError::message(
                    "NumSim TensorMap image has invalid swizzle code",
                ))
            }
        };
        let fill_mode = if flags & (1 << 4) == 0 {
            TensorMapFillMode::Zero
        } else {
            TensorMapFillMode::OobNan
        };
        let host_address = flags & (1 << 5) != 0;
        let allocation_id = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let base_byte_offset = usize::try_from(u64::from_le_bytes(
            bytes[8..16].try_into().unwrap(),
        ))
        .map_err(|_| EngineError::message("TensorMap base byte offset does not fit usize"))?;
        let mut physical_global_shape = [0_usize; 5];
        for (axis, dimension) in physical_global_shape.iter_mut().enumerate() {
            let start = 16 + axis * 4;
            let encoded = u32::from_le_bytes(bytes[start..start + 4].try_into().unwrap());
            let decoded = if encoded == 0 {
                MAX_GLOBAL_DIMENSION
            } else {
                u128::from(encoded)
            };
            *dimension = usize::try_from(decoded).map_err(|_| {
                EngineError::message(format!(
                    "TensorMap global dimension {axis} does not fit usize"
                ))
            })?;
        }
        let mut physical_global_strides = [0_usize; 4];
        let stride_mask = (1_u128 << 36) - 1;
        for pair in 0..2 {
            let start = 36 + pair * 9;
            let mut packed_bytes = [0_u8; 16];
            packed_bytes[..9].copy_from_slice(&bytes[start..start + 9]);
            let packed = u128::from_le_bytes(packed_bytes);
            for item in 0..2 {
                let axis = pair * 2 + item;
                let units = (packed >> (item * 36)) & stride_mask;
                physical_global_strides[axis] = usize::try_from(units << 4).map_err(|_| {
                    EngineError::message(format!(
                        "TensorMap global stride {axis} does not fit usize"
                    ))
                })?;
            }
        }
        let mut box_shape = [1_usize; 5];
        for (axis, dimension) in box_shape.iter_mut().enumerate() {
            *dimension = usize::from(bytes[54 + axis]) + 1;
        }
        let im2col = if bytes[63] & 0x10 != 0 {
            if rank < 3
                || bytes[76] & !7 != 0
                || bytes[77..80].iter().any(|b| *b != 0)
                || box_shape[2..].iter().any(|dimension| *dimension != 1)
            {
                return Err(EngineError::message("invalid im2col TensorMap extension"));
            }
            box_shape[1] += usize::from(bytes[76] & 3) << 8;
            Some(TensorMapIm2col {
                lower: std::array::from_fn(|i| {
                    i16::from_le_bytes(bytes[64 + i * 2..66 + i * 2].try_into().unwrap())
                }),
                upper: std::array::from_fn(|i| {
                    i16::from_le_bytes(bytes[70 + i * 2..72 + i * 2].try_into().unwrap())
                }),
                wide: bytes[76] & 4 != 0,
            })
        } else {
            None
        };
        let encoded_element_strides = u16::from_le_bytes(bytes[61..63].try_into().unwrap());
        let mut element_strides = [1_usize; 5];
        for (axis, stride) in element_strides.iter_mut().enumerate() {
            *stride = usize::from((encoded_element_strides >> (axis * 3)) & 0b111) + 1;
        }
        Ok(Self {
            allocation_id,
            base_byte_offset,
            host_address,
            rank,
            physical_global_shape,
            physical_global_strides,
            box_shape,
            element_strides,
            element_type,
            interleave_bytes,
            fp4_shared_layout,
            swizzle_bytes,
            swizzle_atomicity: SwizzleAtomicity::from_tag(bytes[63]),
            fill_mode,
            im2col,
        })
    }

    /// Recognize a descriptor embedded in an otherwise ordinary byte buffer.
    ///
    /// Direct TensorMap parameters use `read` and report malformed descriptors.
    /// Discovery instead accepts only the exact canonical representation so
    /// arbitrary buffer contents cannot become TensorMaps by sharing the magic
    /// byte and a few valid-looking fields.
    pub(super) fn read_candidate(
        memory: &GlobalMemory,
        view: &BufferView,
    ) -> Result<Option<Self>, EngineError> {
        if view.byte_len() < TENSOR_MAP_DESCRIPTOR_BYTES {
            return Ok(None);
        }
        let bytes = memory.read_bytes(view, 0, TENSOR_MAP_DESCRIPTOR_BYTES)?;
        let payload_bytes = tensor_map_payload_bytes(bytes[63]);
        if bytes[payload_bytes..].iter().any(|byte| *byte != 0) {
            return Ok(None);
        }
        let image = match Self::decode(&bytes[..payload_bytes]) {
            Ok(image) => image,
            Err(_) => return Ok(None),
        };
        // Discovery must agree with bindings._decode_tensor_maps: decoding
        // and re-encoding alone preserves arbitrary inactive fields, so an
        // ordinary scratch buffer can otherwise look like a descriptor.
        // Explicit descriptor reads still use `read`, including while a
        // kernel is replacing individual fields.
        if image.physical_global_shape[image.rank..]
            .iter()
            .any(|dimension| *dimension != 1)
            || image.physical_global_strides[image.rank - 1..]
                .iter()
                .any(|stride| *stride != 0)
            || image.box_shape[image.rank..]
                .iter()
                .any(|dimension| *dimension != 1)
            || image.element_strides[image.rank..]
                .iter()
                .any(|stride| *stride != 1)
            || (image.element_type == TensorMapElementType::Float4E2M1Fn)
                != image.fp4_shared_layout.is_some()
            || (image.interleave_bytes.is_some() && image.rank < 3)
            || (image.swizzle_bytes.is_some_and(|width| width != 128)
                && image.swizzle_atomicity != SwizzleAtomicity::B16)
            || (image.host_address
                && (image.allocation_id == 0 || image.base_byte_offset != 0))
        {
            return Ok(None);
        }
        let canonical = match image.encode() {
            Ok(canonical) => canonical,
            Err(_) => return Ok(None),
        };
        if canonical.as_slice() != &bytes[..payload_bytes] {
            return Ok(None);
        }
        Ok(Some(image))
    }

    pub(super) fn read(
        memory: &GlobalMemory,
        descriptor: &BufferView,
    ) -> Result<Self, EngineError> {
        if descriptor.byte_len() < TENSOR_MAP_DESCRIPTOR_BYTES {
            return Err(EngineError::out_of_bounds(format!(
                "TensorMap descriptor requires {TENSOR_MAP_DESCRIPTOR_BYTES} bytes, but only {} remain",
                descriptor.byte_len()
            )));
        }
        Self::read_payload(|offset, count| Ok(memory.read_bytes(descriptor, offset, count)?))
    }

    pub(super) fn read_payload(
        mut read: impl FnMut(usize, usize) -> Result<Vec<u8>, EngineError>,
    ) -> Result<Self, EngineError> {
        let mut bytes = read(0, TENSOR_MAP_PAYLOAD_BYTES)?;
        let payload_bytes = tensor_map_payload_bytes(bytes[63]);
        if payload_bytes > TENSOR_MAP_PAYLOAD_BYTES {
            bytes.extend(read(
                TENSOR_MAP_PAYLOAD_BYTES,
                payload_bytes - TENSOR_MAP_PAYLOAD_BYTES,
            )?);
        }
        Self::decode(&bytes)
    }

    pub(super) fn write(
        &self,
        memory: &GlobalMemory,
        descriptor: &BufferView,
    ) -> Result<(), EngineError> {
        if descriptor.byte_len() < TENSOR_MAP_DESCRIPTOR_BYTES {
            return Err(EngineError::out_of_bounds(format!(
                "TensorMap descriptor requires {TENSOR_MAP_DESCRIPTOR_BYTES} bytes, but only {} remain",
                descriptor.byte_len()
            )));
        }
        memory.write_bytes(descriptor, 0, &self.encode()?)?;
        Ok(())
    }

    pub(super) fn materialize(
        &self,
        memory: &GlobalMemory,
        expected_rank: usize,
    ) -> Result<RuntimeTensorMap, EngineError> {
        if self.host_address {
            return Err(EngineError::message(
                "NumSim TensorMap host address was not relocated before use",
            ));
        }
        if expected_rank != self.rank {
            return Err(EngineError::message(format!(
                "raw TMA rank {expected_rank} disagrees with TensorMap image rank {}",
                self.rank
            )));
        }
        let full = memory.full_view(AllocationId::from_u64(self.allocation_id))?;
        if self.base_byte_offset > full.byte_len() {
            return Err(EngineError::out_of_bounds(format!(
                "TensorMap base byte offset {} exceeds allocation length {}",
                self.base_byte_offset,
                full.byte_len()
            )));
        }
        let view = memory.subview(
            &full,
            self.base_byte_offset,
            full.byte_len() - self.base_byte_offset,
        )?;
        let global_shape = self.physical_global_shape[..expected_rank].to_vec();
        let mut global_strides =
            self.physical_global_strides[..expected_rank.saturating_sub(1)].to_vec();
        for (axis, stride) in global_strides.iter_mut().enumerate() {
            if *stride == 0 && global_shape[axis + 1] == 1 {
                *stride = 16;
            }
        }
        RuntimeTensorMap::new_with_layout(
            view,
            global_shape,
            global_strides,
            self.box_shape[..if self.im2col.is_some() {
                2
            } else {
                expected_rank
            }]
                .to_vec(),
            self.element_strides[..expected_rank].to_vec(),
            self.element_type.bits(),
            self.element_type,
            self.fp4_shared_layout,
            self.swizzle_bytes,
            self.swizzle_atomicity,
            self.fill_mode,
            self.interleave_bytes,
            self.im2col.clone(),
        )
    }

    pub(super) fn replace_global_address(&mut self, address: BufferView) {
        self.allocation_id = address.allocation().as_u64();
        self.base_byte_offset = address.byte_offset();
        self.host_address = false;
    }

    pub(super) const fn is_host_address(&self) -> bool {
        self.host_address
    }

    pub(super) const fn address_token(&self) -> u64 {
        self.allocation_id
    }

    pub(super) const fn base_byte_offset(&self) -> usize {
        self.base_byte_offset
    }

    pub(super) fn relocate(&mut self, allocation: AllocationId, byte_offset: usize) {
        self.allocation_id = allocation.as_u64();
        self.base_byte_offset = byte_offset;
        self.host_address = false;
    }

    pub(super) fn restore_host_address(&mut self, address: u64) -> Result<(), EngineError> {
        self.allocation_id =
            address
                .checked_add(u64::try_from(self.base_byte_offset).map_err(|_| {
                    EngineError::message("TensorMap base byte offset does not fit u64")
                })?)
                .ok_or_else(|| EngineError::message("TensorMap host address overflow"))?;
        self.base_byte_offset = 0;
        self.host_address = true;
        Ok(())
    }

    pub(super) fn replace_global_dimension(
        &mut self,
        index: usize,
        value: usize,
    ) -> Result<(), EngineError> {
        if index >= self.physical_global_shape.len() || value as u128 >= MAX_GLOBAL_DIMENSION {
            return Err(EngineError::message(format!(
                "TensorMap global dimension field {index}={value} is outside descriptor ranges"
            )));
        }
        self.physical_global_shape[index] = if value == 0 {
            usize::try_from(MAX_GLOBAL_DIMENSION)
                .map_err(|_| EngineError::message("TensorMap dimension 2^32 does not fit usize"))?
        } else {
            value
        };
        Ok(())
    }

    /// PTX field encodings, not CUtensorMapDataType or our private image codes.
    /// Cross-field legality is checked when the updated image is materialized:
    /// several replace instructions may be needed to build a valid new shape.
    pub(super) fn replace_field(
        &mut self,
        field: &str,
        index: Option<usize>,
        value: usize,
    ) -> Result<(), EngineError> {
        match (field, index) {
            ("global_dim", Some(index)) => return self.replace_global_dimension(index, value),
            ("global_stride", Some(index)) => return self.replace_global_stride(index, value),
            ("box_dim" | "element_stride", Some(index)) => {
                let (dimensions, limit) = if field == "box_dim" {
                    (&mut self.box_shape, MAX_BOX_DIMENSION)
                } else {
                    (&mut self.element_strides, MAX_ELEMENT_STRIDE)
                };
                if index >= dimensions.len() || value == 0 || value > limit {
                    return Err(EngineError::message(format!(
                        "TensorMap {field}[{index}]={value} is outside descriptor ranges (ord 0..4, value 1..={limit})"
                    )));
                }
                dimensions[index] = value;
            }
            ("rank", None) if value < 5 => self.rank = value + 1,
            ("interleave_layout", None) if value <= 2 => {
                self.interleave_bytes = match value {
                    0 => None,
                    1 => Some(16),
                    _ => Some(32),
                };
            }
            ("elemtype", None) => {
                self.element_type = match value {
                    0 => TensorMapElementType::U8,
                    1 => TensorMapElementType::U16,
                    2 => TensorMapElementType::U32,
                    3 => TensorMapElementType::I32,
                    4 => TensorMapElementType::U64,
                    5 => TensorMapElementType::I64,
                    6 => TensorMapElementType::F16,
                    7 => TensorMapElementType::F32,
                    8 => TensorMapElementType::F32Ftz,
                    9 => TensorMapElementType::F64,
                    10 => TensorMapElementType::Bf16,
                    11 => TensorMapElementType::Tf32,
                    12 => TensorMapElementType::Tf32Ftz,
                    _ => {
                        return Err(EngineError::message(format!(
                            "TensorMap PTX element type {value} is not modeled"
                        )))
                    }
                };
                // Packed FP4 layout belongs to the old element type.
                self.fp4_shared_layout = None;
            }
            ("fill_mode", None) if value <= 1 => {
                self.fill_mode = if value == 0 {
                    TensorMapFillMode::Zero
                } else {
                    TensorMapFillMode::OobNan
                };
            }
            ("swizzle_mode", None) if value <= 4 => {
                self.swizzle_bytes = match value {
                    0 => None,
                    4 => Some(96),
                    _ => Some(16 << value),
                };
            }
            _ => {
                return Err(EngineError::message(format!(
                    "TensorMap replacement {field}[{index:?}]={value} is not modeled"
                )))
            }
        }
        Ok(())
    }

    pub(super) fn replace_global_stride(
        &mut self,
        index: usize,
        value: usize,
    ) -> Result<(), EngineError> {
        if index >= self.physical_global_strides.len()
            || value as u128 >= MAX_GLOBAL_STRIDE
            || (value != 0 && value % 16 != 0)
        {
            return Err(EngineError::message(format!(
                "TensorMap global stride field {index}={value} is outside descriptor ranges"
            )));
        }
        self.physical_global_strides[index] = value;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RawTmaReductionOp {
    Add,
    Min,
    Max,
    Inc,
    Dec,
    And,
    Or,
    Xor,
}

impl RawTmaReductionOp {
    pub fn resolve(
        self,
        element_type: TensorMapElementType,
    ) -> Result<DeferredGlobalReduction, EngineError> {
        use DeferredGlobalReduction as Reduction;
        use RawTmaReductionOp as Op;

        let reduction = match (self, element_type) {
            (Op::Add, TensorMapElementType::U32) => Reduction::AddU32,
            (Op::Add, TensorMapElementType::I32) => Reduction::AddI32,
            (Op::Add, TensorMapElementType::U64) => Reduction::AddU64,
            (Op::Add, TensorMapElementType::F32 | TensorMapElementType::Tf32) => Reduction::AddF32,
            (Op::Add, TensorMapElementType::F32Ftz | TensorMapElementType::Tf32Ftz) => {
                Reduction::AddF32Ftz
            }
            (Op::Add, TensorMapElementType::F16) => Reduction::AddF16,
            (Op::Add, TensorMapElementType::Bf16) => Reduction::AddBf16,
            (Op::Min, TensorMapElementType::U32) => Reduction::MinU32,
            (Op::Min, TensorMapElementType::I32) => Reduction::MinI32,
            (Op::Min, TensorMapElementType::U64) => Reduction::MinU64,
            (Op::Min, TensorMapElementType::I64) => Reduction::MinI64,
            (Op::Min, TensorMapElementType::F16) => Reduction::MinF16,
            (Op::Min, TensorMapElementType::Bf16) => Reduction::MinBf16,
            (Op::Max, TensorMapElementType::U32) => Reduction::MaxU32,
            (Op::Max, TensorMapElementType::I32) => Reduction::MaxI32,
            (Op::Max, TensorMapElementType::U64) => Reduction::MaxU64,
            (Op::Max, TensorMapElementType::I64) => Reduction::MaxI64,
            (Op::Max, TensorMapElementType::F16) => Reduction::MaxF16,
            (Op::Max, TensorMapElementType::Bf16) => Reduction::MaxBf16,
            (Op::Inc, TensorMapElementType::U32) => Reduction::IncU32,
            (Op::Dec, TensorMapElementType::U32) => Reduction::DecU32,
            (Op::And, element_type) if element_type.bits() == 32 => Reduction::AndB32,
            (Op::And, element_type) if element_type.bits() == 64 => Reduction::AndB64,
            (Op::Or, element_type) if element_type.bits() == 32 => Reduction::OrB32,
            (Op::Or, element_type) if element_type.bits() == 64 => Reduction::OrB64,
            (Op::Xor, element_type) if element_type.bits() == 32 => Reduction::XorB32,
            (Op::Xor, element_type) if element_type.bits() == 64 => Reduction::XorB64,
            _ => {
                return Err(EngineError::message(format!(
                    "cp.reduce.async.bulk.tensor operation {self:?} is invalid for TensorMap dtype {element_type}"
                )));
            }
        };
        Ok(reduction)
    }
}

impl RuntimeTensorMap {
    /// Data allocation bound by the host's immutable TensorMap parameter.
    /// Stores through writable descriptors retain a conservative write seed.
    pub fn allocation(&self) -> AllocationId {
        self.view.allocation()
    }

    fn read_global_bytes(
        &self,
        physical: &PhysicalMemory,
        byte_offset: usize,
        output: &mut [u8],
    ) -> Result<(), EngineError> {
        physical
            .global()
            .read_bytes_into(&self.view, byte_offset, output)?;
        Ok(())
    }

    /// Build instruction-local metadata without modifying the source descriptor.
    pub(crate) fn with_overrides(
        &self,
        memory: &GlobalMemory,
        address: BufferView,
        dimensions: &[i64],
        lower_strides: &[i64],
        upper_strides: i64,
        coordinates: &[i64],
    ) -> Result<Self, EngineError> {
        let rank = self.rank();
        let observed_base = address.observed_allocation_address().unwrap_or(0);
        if (observed_base % 16 + (address.byte_offset() % 16) as u64) % 16 != 0 {
            return Err(EngineError::message(
                "TMA override address must be 16-byte aligned",
            ));
        }
        if address.byte_len() < 128 * 1024 {
            return Err(EngineError::out_of_bounds(
                "TMA override address requires 128 KiB of accessible memory",
            ));
        }
        let mut image = RuntimeTensorMapImage::from_tensor_map(self);
        image.replace_global_address(address);
        if dimensions.is_empty() {
            if !lower_strides.is_empty() || upper_strides != 0 {
                return Err(EngineError::message(
                    "TMA stride override requires dimension override",
                ));
            }
        } else {
            if dimensions.len() != rank
                || lower_strides.len() != rank - 1
                || coordinates.len() != rank
            {
                return Err(EngineError::message(
                    "TMA override rank/operand counts disagree",
                ));
            }
            if coordinates.iter().any(|coordinate| *coordinate != 0) {
                return Err(EngineError::message(
                    "TMA attribute override requires zero coordinates",
                ));
            }
            for (axis, &dimension) in dimensions.iter().enumerate() {
                if !(1..=255).contains(&dimension) {
                    return Err(EngineError::message(
                        "TMA override dimension must be a nonzero 8-bit value",
                    ));
                }
                image.replace_global_dimension(axis, dimension as usize)?;
            }
            let upper = u16::try_from(upper_strides)
                .map_err(|_| EngineError::message("TMA upper strides must fit 16 bits"))?;
            if u32::from(upper) >> (4 * (rank - 1)) != 0 {
                return Err(EngineError::message(
                    "TMA upper strides have nonzero unused bits",
                ));
            }
            for (axis, &lower) in lower_strides.iter().enumerate() {
                let lower = u32::try_from(lower)
                    .map_err(|_| EngineError::message("TMA lower stride must fit 32 bits"))?;
                let high = u64::from((upper >> (4 * axis)) & 15);
                let stride = usize::try_from((u64::from(lower) | (high << 32)) << 4)
                    .map_err(|_| EngineError::message("TMA stride does not fit usize"))?;
                image.replace_global_stride(axis, stride)?;
            }
        }
        image.materialize(memory, rank)
    }

    pub(crate) const fn rank(&self) -> usize {
        self.global_shape.len()
    }

    fn transfer_element_bits(&self) -> usize {
        self.interleave_bytes
            .map_or(self.element_bits, |bytes| bytes * 8)
    }

    fn validate_swizzle_direction(&self, load: bool) -> Result<(), EngineError> {
        // The descriptor is usable by cache hints and predicated-off copies.
        // Reject the unresolved shared layout only when a transfer consumes it.
        if self.interleave_bytes == Some(16) && self.swizzle_bytes.is_some() {
            return Err(EngineError::analysis_incomplete(
                "tma_swizzled_16b_interleave_unmodeled",
            ));
        }
        if self.swizzle_bytes.is_none() || self.swizzle_atomicity == SwizzleAtomicity::B16 {
            return Ok(());
        }
        if !load && self.swizzle_atomicity == SwizzleAtomicity::B32Flip8 {
            return Err(EngineError::message(
                "8B flip is only valid for global-to-shared tensor copies",
            ));
        }
        if load && self.swizzle_atomicity == SwizzleAtomicity::B64 {
            if self.element_type == TensorMapElementType::U6
                || self.fp4_shared_layout == Some(Fp4SharedLayout::Align16Padded)
            {
                return Err(EngineError::message(
                    "64B atomicity loads are invalid for U6 and padded FP4 TensorMaps",
                ));
            }
            // The independent SM100 load probe traps; store is GPU-validated.
            return Err(EngineError::analysis_incomplete(
                "tma_64b_atomicity_load_unmodeled",
            ));
        }
        Ok(())
    }

    fn validate_u6_origin(&self, origin: &[i64]) -> Result<(), EngineError> {
        if self.element_bits == 6
            && origin.first().is_some_and(|value| value.rem_euclid(128) != 0)
        {
            return Err(EngineError::message(
                "SM100 U6 TensorMap inner origin must be a multiple of 128",
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        view: BufferView,
        global_shape: Vec<usize>,
        global_strides: Vec<usize>,
        box_shape: Vec<usize>,
        element_strides: Vec<usize>,
        element_bits: usize,
        element_type: TensorMapElementType,
        fp4_shared_layout: Option<Fp4SharedLayout>,
        swizzle_bytes: Option<usize>,
        fill_mode: TensorMapFillMode,
    ) -> Result<Self, EngineError> {
        Self::new_with_interleave(
            view,
            global_shape,
            global_strides,
            box_shape,
            element_strides,
            element_bits,
            element_type,
            fp4_shared_layout,
            swizzle_bytes,
            fill_mode,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_interleave(
        view: BufferView,
        global_shape: Vec<usize>,
        global_strides: Vec<usize>,
        box_shape: Vec<usize>,
        element_strides: Vec<usize>,
        element_bits: usize,
        element_type: TensorMapElementType,
        fp4_shared_layout: Option<Fp4SharedLayout>,
        swizzle_bytes: Option<usize>,
        fill_mode: TensorMapFillMode,
        interleave_bytes: Option<usize>,
    ) -> Result<Self, EngineError> {
        Self::new_with_layout(
            view,
            global_shape,
            global_strides,
            box_shape,
            element_strides,
            element_bits,
            element_type,
            fp4_shared_layout,
            swizzle_bytes,
            SwizzleAtomicity::B16,
            fill_mode,
            interleave_bytes,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_layout(
        view: BufferView,
        global_shape: Vec<usize>,
        global_strides: Vec<usize>,
        box_shape: Vec<usize>,
        mut element_strides: Vec<usize>,
        element_bits: usize,
        element_type: TensorMapElementType,
        fp4_shared_layout: Option<Fp4SharedLayout>,
        swizzle_bytes: Option<usize>,
        swizzle_atomicity: SwizzleAtomicity,
        fill_mode: TensorMapFillMode,
        interleave_bytes: Option<usize>,
        im2col: Option<TensorMapIm2col>,
    ) -> Result<Self, EngineError> {
        let rank = global_shape.len();
        if rank == 0
            || rank > 5
            || global_strides.len() + 1 != rank
            || box_shape.len() != if im2col.is_some() { 2 } else { rank }
            || element_strides.len() != rank
            || global_shape.contains(&0)
            || box_shape.contains(&0)
        {
            return Err(EngineError::message(
                "TensorMap rank/shape/stride metadata is inconsistent",
            ));
        }
        if swizzle_bytes.is_some_and(|width| width != 128)
            && swizzle_atomicity != SwizzleAtomicity::B16
        {
            return Err(EngineError::message(
                "non-default swizzle atomicity requires 128B swizzle",
            ));
        }
        if swizzle_atomicity == SwizzleAtomicity::B32Flip8
            && (element_type == TensorMapElementType::U6
                || fp4_shared_layout == Some(Fp4SharedLayout::Align16Padded))
        {
            return Err(EngineError::message(
                "padded FP4 and U6 TensorMaps do not support 8B flip",
            ));
        }
        if im2col.as_ref().is_some_and(|config| config.wide)
            && matches!(
                swizzle_atomicity,
                SwizzleAtomicity::B32Flip8 | SwizzleAtomicity::B64
            )
        {
            return Err(EngineError::message(
                "wide im2col does not support 8B flip or 64B atomicity",
            ));
        }
        let dtype_bits = element_type.bits();
        if dtype_bits != element_bits {
            return Err(EngineError::message(format!(
                "TensorMap logical dtype {element_type} has {dtype_bits} bits, but descriptor declares {element_bits} bits"
            )));
        }
        if global_shape
            .iter()
            .any(|dimension| *dimension as u128 > MAX_GLOBAL_DIMENSION)
        {
            return Err(EngineError::message(
                "TensorMap global dimensions must be at most 2^32",
            ));
        }
        if im2col.is_none()
            && box_shape
                .iter()
                .any(|dimension| *dimension > MAX_BOX_DIMENSION)
        {
            return Err(EngineError::message(
                "TensorMap box dimensions must be in 1..=256",
            ));
        }
        if let Some(config) = &im2col {
            if rank < 3 || box_shape[0] > 256 || box_shape[1] > 1024 {
                return Err(EngineError::message(
                    "im2col requires rank 3..5, 1..256 channels and 1..1024 pixels",
                ));
            }
            let spatial_rank = if config.wide { 1 } else { rank - 2 };
            let bits = if config.wide || rank == 3 {
                16
            } else if rank == 4 {
                8
            } else {
                5
            };
            for axis in 0..3 {
                let lo = i64::from(config.lower[axis]);
                let hi = i64::from(config.upper[axis]);
                if axis >= spatial_rank {
                    if lo != 0 || hi != 0 {
                        return Err(EngineError::message("im2col has nonzero unused corners"));
                    }
                } else if lo < -(1 << (bits - 1))
                    || hi < -(1 << (bits - 1))
                    || lo >= (1 << (bits - 1))
                    || hi >= (1 << (bits - 1))
                    || lo
                        >= global_shape[axis + usize::from(interleave_bytes.is_none())] as i64 + hi
                {
                    return Err(EngineError::message(
                        "im2col has invalid bounding-box corners",
                    ));
                }
            }
            if config.wide && interleave_bytes.is_some() {
                return Err(EngineError::message(
                    "wide im2col does not support interleave",
                ));
            }
            if config.wide && !matches!(swizzle_bytes, Some(64 | 96 | 128)) {
                return Err(EngineError::message(
                    "wide im2col requires 64B, 96B or 128B swizzle",
                ));
            }
        }
        let observed_base = view.observed_allocation_address().unwrap_or(0);
        let address_aligned = |alignment: u64| {
            (observed_base % alignment + (view.byte_offset() as u64 % alignment)) % alignment == 0
        };
        if let Some(bytes) = interleave_bytes {
            if swizzle_bytes == Some(96) {
                return Err(EngineError::message(
                    "96B swizzle does not support interleave",
                ));
            }
            if !matches!(bytes, 16 | 32) || rank < 3 {
                return Err(EngineError::message(
                    "TensorMap interleave requires rank 3..=5 and 16B or 32B slices",
                ));
            }
            if element_strides[0] == 0 {
                return Err(EngineError::message(
                    "interleaved TensorMap axis-zero stride must be nonzero",
                ));
            }
            if bytes == 32
                && (swizzle_bytes != Some(32)
                    || !address_aligned(32)
                    || global_strides.iter().any(|s| s % 32 != 0))
            {
                return Err(EngineError::message(
                    "32B interleave requires 32B swizzle, address and strides",
                ));
            }
        }
        // Without interleave hardware ignores axis zero, including its zero sentinel.
        if element_strides[0] > MAX_ELEMENT_STRIDE
            || element_strides[1..]
                .iter()
                .any(|stride| *stride == 0 || *stride > MAX_ELEMENT_STRIDE)
        {
            return Err(EngineError::message(
                "TensorMap element strides must be in 1..=8 (axis zero may be zero when interleave is disabled)",
            ));
        }
        if interleave_bytes.is_none() {
            element_strides[0] = 1;
        }
        if global_strides
            .iter()
            .any(|stride| *stride == 0 || *stride as u128 >= MAX_GLOBAL_STRIDE || stride % 16 != 0)
        {
            return Err(EngineError::message(
                "TensorMap global byte strides must be non-zero multiples of 16 below 2^40",
            ));
        }
        match (element_bits, fp4_shared_layout) {
            (4, Some(_)) | (6 | 8 | 16 | 32 | 64, None) => {}
            (4, None) => {
                return Err(EngineError::message(
                    "FP4 TensorMap is missing its shared layout",
                ));
            }
            (_, Some(layout)) => {
                return Err(EngineError::message(format!(
                    "non-FP4 TensorMap cannot use FP4 shared layout {layout:?}"
                )));
            }
            (_, None) => {
                return Err(EngineError::message(format!(
                    "TensorMap has unsupported {element_bits}-bit elements"
                )));
            }
        }
        if fill_mode == TensorMapFillMode::OobNan && !element_type.supports_oob_nan() {
            return Err(EngineError::message(
                "TensorMap OOB-NaN fill requires a 16/32/64-bit floating-point dtype",
            ));
        }
        if swizzle_bytes.is_some_and(|bytes| !matches!(bytes, 32 | 64 | 96 | 128)) {
            return Err(EngineError::message(
                "TensorMap swizzle must be 32, 64, 96, or 128 bytes",
            ));
        }
        if !address_aligned(16) {
            return Err(EngineError::message(
                "TensorMap global address must be 16-byte aligned",
            ));
        }
        if element_bits == 6 {
            if interleave_bytes.is_some() {
                return Err(EngineError::message(
                    "U6 TensorMap does not support interleave",
                ));
            }
            if !global_shape[0].is_multiple_of(128) || box_shape[0] != 128 {
                return Err(EngineError::message(
                    "SM100 U6 TensorMap requires dimension zero in multiples of 128 and box zero 128",
                ));
            }
            if !address_aligned(32) || global_strides.iter().any(|stride| stride % 32 != 0) {
                return Err(EngineError::message(
                    "SM100 U6 TensorMap address and strides must be 32-byte aligned",
                ));
            }
            if swizzle_bytes.is_some_and(|bytes| bytes != 128) {
                return Err(EngineError::message(
                    "SM100 U6 TensorMap supports only none or 128B swizzle",
                ));
            }
        }
        match fp4_shared_layout {
            Some(Fp4SharedLayout::Align8Packed) => {
                if !global_shape[0].is_multiple_of(2) {
                    return Err(EngineError::message(
                        "align8 packed FP4 TensorMap global dimension zero must be a multiple of 2",
                    ));
                }
            }
            Some(Fp4SharedLayout::Align16Padded) => {
                // Interleaved padded dimensions do not yet have a validated
                // unit contract; do not apply the non-interleaved 128 rule.
                if interleave_bytes.is_some() {
                    return Err(EngineError::analysis_incomplete(
                        "tma_padded_fp4_interleave_unmodeled",
                    ));
                }
                if !global_shape[0].is_multiple_of(128) || box_shape[0] != 128 {
                    return Err(EngineError::message(
                        "align16 padded FP4 TensorMap requires global dimension zero to be a multiple of 128 and box dimension zero to equal 128",
                    ));
                }
                if !address_aligned(32) || global_strides.iter().any(|stride| stride % 32 != 0) {
                    return Err(EngineError::message(
                        "align16 padded FP4 TensorMap address and global strides must be 32-byte aligned",
                    ));
                }
                if swizzle_bytes.is_some_and(|bytes| bytes != 128) {
                    return Err(EngineError::message(
                        "align16 padded FP4 TensorMap only supports 128B swizzle",
                    ));
                }
            }
            None => {}
        }

        let transfer_bits = interleave_bytes.map_or(element_bits, |bytes| bytes * 8);
        let inner_global_bytes = global_shape[0]
            .checked_mul(transfer_bits)
            .and_then(|bits| bits.checked_add(7))
            .map(|bits| bits / 8)
            .ok_or_else(|| EngineError::message("TensorMap inner global size overflow"))?;
        let mut varying_outer_axes = global_strides
            .iter()
            .copied()
            .zip(global_shape.iter().copied().skip(1))
            .enumerate()
            .filter_map(|(axis, (stride, dimension))| {
                (dimension > 1).then_some((stride, dimension, axis))
            })
            .collect::<Vec<_>>();
        varying_outer_axes.sort_unstable_by_key(|(stride, _, axis)| (*stride, *axis));
        let mut occupied_span = inner_global_bytes;
        for (stride, dimension, axis) in varying_outer_axes {
            if stride < occupied_span {
                return Err(EngineError::message(format!(
                    "TensorMap global stride {axis} is {stride} bytes, which overlaps the prior {occupied_span}-byte span"
                )));
            }
            occupied_span = (dimension - 1)
                .checked_mul(stride)
                .and_then(|axis_span| occupied_span.checked_add(axis_span))
                .ok_or_else(|| EngineError::message("TensorMap global span overflow"))?;
        }
        let required_byte_len = global_strides
            .iter()
            .zip(global_shape.iter().skip(1))
            .try_fold(inner_global_bytes, |span, (stride, dimension)| {
                span.checked_add(
                    (dimension - 1)
                        .checked_mul(*stride)
                        .ok_or_else(|| EngineError::message("TensorMap global span overflow"))?,
                )
                .ok_or_else(|| EngineError::message("TensorMap global span overflow"))
            })?;
        if view.byte_len() < required_byte_len {
            return Err(EngineError::message(format!(
                "TensorMap requires {required_byte_len} global bytes, but its view has {}",
                view.byte_len()
            )));
        }

        let descriptor_inner_bytes = box_shape[0]
            .checked_mul(transfer_bits)
            .and_then(|bits| bits.checked_add(7))
            .map(|bits| bits / 8)
            .ok_or_else(|| EngineError::message("TensorMap inner box size overflow"))?;
        if descriptor_inner_bytes % 16 != 0 {
            return Err(EngineError::message(
                "TensorMap inner box transfer size must be a multiple of 16 bytes without interleave",
            ));
        }
        let shared_inner_bytes = if element_bits == 6
            || fp4_shared_layout == Some(Fp4SharedLayout::Align16Padded)
        {
            box_shape[0]
        } else {
            descriptor_inner_bytes
        };
        if let Some(swizzle_bytes) = swizzle_bytes.filter(|_| interleave_bytes.is_none()) {
            if shared_inner_bytes > swizzle_bytes {
                return Err(EngineError::message(format!(
                    "TensorMap row uses {shared_inner_bytes} bytes, exceeding {swizzle_bytes}B swizzle"
                )));
            }
        }

        let mut traversal_shape = if im2col.is_some() {
            box_shape.clone()
        } else {
            box_shape
                .iter()
                .zip(&element_strides)
                .map(|(dimension, stride)| dimension.div_ceil(*stride))
                .collect::<Vec<_>>()
        };
        if interleave_bytes.is_some() && im2col.is_none() {
            // Interleaved descriptors traverse spatial axes and N; the
            // penultimate coordinate selects one channel slice.
            traversal_shape[rank - 2] = 1;
        }
        if interleave_bytes.is_some() && im2col.is_some() {
            // Each im2col pixel selects exactly one channel slice; hardware
            // ignores channels-per-pixel in an interleaved descriptor.
            traversal_shape[0] = 1;
        }
        // Im2col selects pixel origins at issue time; the ordinary template
        // remains the sole owner of channel packing and global OOB handling.
        let template_shape = if im2col.is_some() {
            let mut shape = vec![1; rank];
            shape[0] = traversal_shape[0];
            shape
        } else {
            traversal_shape.clone()
        };
        let transfer_template = Arc::new(TensorMapTransferTemplate::compile(
            &template_shape,
            &element_strides,
            &global_strides,
            transfer_bits,
            fp4_shared_layout,
            swizzle_bytes,
            swizzle_atomicity,
        )?);
        let mut physical_global_shape = [1_usize; 5];
        physical_global_shape[..rank].copy_from_slice(&global_shape);
        let mut physical_global_strides = [0_usize; 4];
        physical_global_strides[..global_strides.len()].copy_from_slice(&global_strides);
        Ok(Self {
            view,
            global_shape,
            global_strides,
            physical_global_shape,
            physical_global_strides,
            box_shape,
            element_strides,
            traversal_shape,
            element_bits,
            interleave_bytes,
            element_type,
            fp4_shared_layout,
            swizzle_bytes,
            swizzle_atomicity,
            fill_mode,
            transfer_template,
            im2col,
        })
    }

    fn reduction(
        &self,
        operation: RawTmaReductionOp,
    ) -> Result<DeferredGlobalReduction, EngineError> {
        operation.resolve(self.element_type)
    }

    fn im2col_origins(
        &self,
        origin: &[i64],
        mode: Im2colMode,
        info: &[i64],
    ) -> Result<Vec<Vec<i64>>, EngineError> {
        let layout = self.im2col.as_ref().ok_or_else(|| {
            EngineError::message("im2col instruction requires an im2col TensorMap")
        })?;
        let rank = self.rank();
        let wide = mode != Im2colMode::Spatial;
        if origin.len() != rank || wide != layout.wide {
            return Err(EngineError::message(
                "im2col instruction/descriptor layout mismatch",
            ));
        }
        let spatial = if wide { 1 } else { rank - 2 };
        let first_spatial = usize::from(self.interleave_bytes.is_none());
        let (halo, shift) = if wide {
            if info.len() != 2
                || !(0..32).contains(&info[1])
                || !(0..if mode == Im2colMode::Wide128 { 32 } else { 512 }).contains(&info[0])
            {
                return Err(EngineError::message("invalid im2col wHalo/wOffset"));
            }
            (info[0] as usize, info[1])
        } else {
            let bits = match rank {
                3 => 16,
                4 => 8,
                _ => 5,
            };
            if info.len() != spatial || info.iter().any(|value| !(0..1_i64 << bits).contains(value))
            {
                return Err(EngineError::message("invalid im2col spatial offsets"));
            }
            (0, 0)
        };
        let pixels = if mode == Im2colMode::Wide128 {
            128
        } else {
            self.box_shape[1]
        };
        if mode == Im2colMode::Wide128 && self.swizzle_bytes == Some(96) {
            return Err(EngineError::message(
                "im2col::w::128 does not support 96B swizzle",
            ));
        }
        let lower = (0..spatial)
            .map(|axis| i64::from(layout.lower[axis]) + shift)
            .collect::<Vec<_>>();
        let upper = (0..spatial)
            .map(|axis| {
                self.global_shape[axis + first_spatial] as i64
                    + i64::from(layout.upper[axis])
                    + shift
            })
            .collect::<Vec<_>>();
        let mut cursor = origin.to_vec();
        cursor[first_spatial] = cursor[first_spatial]
            .checked_add(shift)
            .ok_or_else(|| EngineError::message("im2col coordinate overflow"))?;
        if (0..spatial).any(|axis| {
            (!wide && cursor[axis + first_spatial] < lower[axis])
                || cursor[axis + first_spatial] >= upper[axis]
        }) {
            return Err(EngineError::message(
                "im2col filter origin is outside its traversal bounds",
            ));
        }
        let advance = |cursor: &mut Vec<i64>| -> Result<(), EngineError> {
            for axis in 0..spatial {
                cursor[axis + first_spatial] = cursor[axis + first_spatial]
                    .checked_add(self.element_strides[axis + first_spatial] as i64)
                    .ok_or_else(|| EngineError::message("im2col coordinate overflow"))?;
                if cursor[axis + first_spatial] < upper[axis] {
                    return Ok(());
                }
                cursor[axis + first_spatial] = lower[axis];
            }
            cursor[rank - 1] = cursor[rank - 1]
                .checked_add(self.element_strides[rank - 1] as i64)
                .ok_or_else(|| EngineError::message("im2col batch coordinate overflow"))?;
            Ok(())
        };
        let chunks = if mode == Im2colMode::Wide128 { 4 } else { 1 };
        let mut origins = vec![Vec::new(); pixels + chunks * halo];
        for pixel in 0..pixels {
            let mut address = cursor.clone();
            if !wide {
                for axis in 0..spatial {
                    address[axis + first_spatial] = address[axis + first_spatial]
                        .checked_add(info[axis])
                        .ok_or_else(|| EngineError::message("im2col offset overflow"))?;
                }
            }
            origins[pixel] = address;
            advance(&mut cursor)?;
            if (pixel + 1) % (pixels / chunks) == 0 {
                let chunk = (pixel + 1) / (pixels / chunks) - 1;
                let mut halo_cursor = cursor.clone();
                for index in 0..halo {
                    // w::128 appends an interleaved halo plane after its
                    // 128 main pixels, rather than appending to each chunk.
                    origins[pixels + index * chunks + chunk] = halo_cursor.clone();
                    advance(&mut halo_cursor)?;
                }
            }
        }
        Ok(origins)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TensorMapGeometry {
    packed_elements: usize,
    unit_bytes: usize,
    unit_stride_bytes: usize,
    inner_units: usize,
    inner_row_bytes: usize,
    outer_count: usize,
}

fn tensor_map_geometry_from_metadata(
    traversal_shape: &[usize],
    element_bits: usize,
    fp4_shared_layout: Option<Fp4SharedLayout>,
) -> Result<TensorMapGeometry, EngineError> {
    let (packed_elements, unit_bytes, unit_stride_bytes) = match (element_bits, fp4_shared_layout) {
        (4, Some(Fp4SharedLayout::Align8Packed)) => (2_usize, 1_usize, 1_usize),
        (4, Some(Fp4SharedLayout::Align16Padded)) => (16_usize, 8_usize, 16_usize),
        (6, None) => (16_usize, 12_usize, 16_usize),
        (4, None) => {
            return Err(EngineError::message(
                "FP4 TensorMap is missing its shared layout",
            ));
        }
        (8 | 16 | 32 | 64 | 128 | 256, None) | (128 | 256, Some(Fp4SharedLayout::Align8Packed)) => {
            // Interleave transfers whole byte slices even for packed FP4;
            // only non-interleaved four-bit units need nibble processing.
            let bytes = element_bits / 8;
            (1_usize, bytes, bytes)
        }
        (bits, layout) => {
            return Err(EngineError::message(format!(
                "unsupported TensorMap geometry for {bits}-bit elements and {layout:?}"
            )));
        }
    };
    let inner_units = traversal_shape[0]
        .checked_add(packed_elements - 1)
        .ok_or_else(|| EngineError::message("TensorMap inner box size overflow"))?
        / packed_elements;
    let inner_row_bytes = inner_units
        .checked_mul(unit_stride_bytes)
        .ok_or_else(|| EngineError::message("TensorMap inner row byte size overflow"))?;
    let outer_count = traversal_shape[1..]
        .iter()
        .try_fold(1_usize, |product, extent| {
            product
                .checked_mul(*extent)
                .ok_or_else(|| EngineError::message("TensorMap outer box size overflow"))
        })?;
    Ok(TensorMapGeometry {
        packed_elements,
        unit_bytes,
        unit_stride_bytes,
        inner_units,
        inner_row_bytes,
        outer_count,
    })
}

fn tensor_map_geometry(tensor_map: &RuntimeTensorMap) -> Result<TensorMapGeometry, EngineError> {
    if tensor_map.global_shape.is_empty()
        || tensor_map.box_shape.len()
            != if tensor_map.im2col.is_some() {
                2
            } else {
                tensor_map.rank()
            }
        || tensor_map.global_shape.len() != tensor_map.element_strides.len()
        || tensor_map.global_strides.len() + 1 != tensor_map.global_shape.len()
    {
        return Err(EngineError::message(
            "TensorMap rank metadata is inconsistent",
        ));
    }
    tensor_map_geometry_from_metadata(
        &tensor_map.traversal_shape,
        tensor_map.transfer_element_bits(),
        tensor_map.fp4_shared_layout,
    )
}

fn tensor_map_outer_coordinates(tensor_map: &RuntimeTensorMap, mut linear: usize) -> Vec<usize> {
    let mut coordinates = vec![0_usize; tensor_map.traversal_shape.len()];
    for (axis, coordinate) in coordinates.iter_mut().enumerate().skip(1) {
        *coordinate = linear % tensor_map.traversal_shape[axis];
        linear /= tensor_map.traversal_shape[axis];
    }
    coordinates
}

fn tensor_map_global_coordinates(
    tensor_map: &RuntimeTensorMap,
    origin: &[i64],
    inner_element: usize,
    outer: &[usize],
) -> Result<Vec<i64>, EngineError> {
    if origin.len() != tensor_map.global_shape.len() {
        return Err(EngineError::message(format!(
            "TensorMap expected {} coordinates, got {}",
            tensor_map.global_shape.len(),
            origin.len()
        )));
    }
    let mut result = vec![0_i64; origin.len()];
    tensor_map_global_coordinates_into(tensor_map, origin, inner_element, outer, &mut result)?;
    Ok(result)
}

fn tensor_map_global_coordinates_into(
    tensor_map: &RuntimeTensorMap,
    origin: &[i64],
    inner_element: usize,
    outer: &[usize],
    result: &mut [i64],
) -> Result<(), EngineError> {
    if origin.len() != tensor_map.global_shape.len() {
        return Err(EngineError::message(format!(
            "TensorMap expected {} coordinates, got {}",
            tensor_map.global_shape.len(),
            origin.len()
        )));
    }
    if result.len() != origin.len() {
        return Err(EngineError::message(format!(
            "TensorMap coordinate scratch has {} entries, expected {}",
            result.len(),
            origin.len()
        )));
    }
    for axis in 0..origin.len() {
        let local = if axis == 0 {
            inner_element
        } else {
            outer[axis]
        };
        let delta = local
            .checked_mul(tensor_map.element_strides[axis])
            .and_then(|value| i64::try_from(value).ok())
            .ok_or_else(|| EngineError::message("TensorMap coordinate delta overflow"))?;
        result[axis] = origin[axis]
            .checked_add(delta)
            .ok_or_else(|| EngineError::message("TensorMap coordinate overflow"))?;
    }
    Ok(())
}

fn tensor_map_coordinates_in_bounds(tensor_map: &RuntimeTensorMap, coordinates: &[i64]) -> bool {
    coordinates
        .iter()
        .zip(&tensor_map.global_shape)
        .all(|(coordinate, extent)| {
            *coordinate >= 0 && usize::try_from(*coordinate).is_ok_and(|value| value < *extent)
        })
}

fn tensor_map_global_byte_offset(
    tensor_map: &RuntimeTensorMap,
    coordinates: &[i64],
) -> Result<(usize, usize), EngineError> {
    if !tensor_map_coordinates_in_bounds(tensor_map, coordinates) {
        return Err(EngineError::message(
            "TensorMap coordinate is outside global shape",
        ));
    }
    let inner = usize::try_from(coordinates[0])
        .map_err(|_| EngineError::message("negative TensorMap inner coordinate"))?;
    let inner_bits = inner
        .checked_mul(tensor_map.transfer_element_bits())
        .ok_or_else(|| EngineError::message("TensorMap inner bit offset overflow"))?;
    let mut byte_offset = inner_bits / 8;
    for (axis, coordinate) in coordinates.iter().enumerate().skip(1) {
        let coordinate = usize::try_from(*coordinate)
            .map_err(|_| EngineError::message("negative TensorMap coordinate"))?;
        byte_offset = byte_offset
            .checked_add(
                coordinate
                    .checked_mul(tensor_map.global_strides[axis - 1])
                    .ok_or_else(|| EngineError::message("TensorMap stride offset overflow"))?,
            )
            .ok_or_else(|| EngineError::message("TensorMap byte offset overflow"))?;
    }
    Ok((byte_offset, inner_bits % 8))
}

fn tensor_map_shared_byte_offset(
    tensor_map: &RuntimeTensorMap,
    outer_linear: usize,
    inner_byte: usize,
    inner_row_bytes: usize,
    absolute_base: usize,
) -> Result<usize, EngineError> {
    tensor_map_shared_byte_offset_from_layout(
        tensor_map.swizzle_bytes,
        tensor_map.swizzle_atomicity,
        outer_linear,
        inner_byte,
        inner_row_bytes,
        absolute_base,
    )
}

fn tensor_map_shared_byte_offset_from_layout(
    swizzle_bytes: Option<usize>,
    atomicity: SwizzleAtomicity,
    outer_linear: usize,
    inner_byte: usize,
    inner_row_bytes: usize,
    absolute_base: usize,
) -> Result<usize, EngineError> {
    let Some(swizzle_bytes) = swizzle_bytes else {
        return outer_linear
            .checked_mul(inner_row_bytes)
            .and_then(|row| row.checked_add(inner_byte))
            .ok_or_else(|| EngineError::message("dense TensorMap shared offset overflow"));
    };
    if swizzle_bytes == 96 {
        // PTX Figure 32 packs 96-byte rows consecutively, then exchanges
        // adjacent 16-byte atoms in every odd 128-byte band. Unlike 32/64/128
        // modes, this is not a power-of-two row stride/XOR group count.
        let absolute = outer_linear
            .checked_mul(96)
            .and_then(|row| row.checked_add(inner_byte))
            .and_then(|offset| absolute_base.checked_add(offset))
            .ok_or_else(|| EngineError::message("96B swizzle address overflow"))?;
        return (absolute ^ ((absolute >> 3) & 16))
            .checked_sub(absolute_base)
            .ok_or_else(|| EngineError::message("96B swizzle precedes shared base"));
    }
    let atom_bytes = atomicity.bytes();
    let groups = swizzle_bytes / atom_bytes;
    if !matches!(groups, 2 | 4 | 8) {
        return Err(EngineError::message(format!(
            "TensorMap swizzle has unsupported atom group count {groups}"
        )));
    }
    // Ordinary rows occupy at least one swizzle span. Interleaved rows can
    // span several groups; both use the same absolute-address atom XOR.
    let absolute = outer_linear
        .checked_mul(inner_row_bytes.max(swizzle_bytes))
        .and_then(|row| row.checked_add(inner_byte))
        .and_then(|offset| absolute_base.checked_add(offset))
        .ok_or_else(|| EngineError::message("swizzled TensorMap shared offset overflow"))?;
    let flipped = if atomicity == SwizzleAtomicity::B32Flip8 {
        absolute ^ ((absolute >> 4) & 8)
    } else {
        absolute
    };
    (flipped ^ (((absolute >> 7) & (groups - 1)) * atom_bytes))
        .checked_sub(absolute_base)
        .ok_or_else(|| EngineError::message("swizzled TensorMap precedes shared base"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TensorMapByteRun {
    byte_offset: usize,
    payload_offset: usize,
    byte_len: usize,
}

#[derive(Clone, Debug)]
struct TensorMapSharedRunProgram {
    runs: Arc<[TensorMapByteRun]>,
    byte_extent: usize,
}

#[derive(Debug)]
struct TensorMapBoundSharedProgram {
    runs: Vec<TensorMapByteRun>,
}

#[derive(Clone, Debug)]
struct TensorMapRowTemplate {
    coordinate_deltas: Box<[i64]>,
    global_outer_byte_delta: usize,
    payload_offset: usize,
}

#[derive(Clone, Debug)]
struct TensorMapTransferTemplate {
    geometry: TensorMapGeometry,
    rows: Arc<[TensorMapRowTemplate]>,
    source_row_order: Arc<[usize]>,
    max_coordinate_deltas: Box<[i64]>,
    shared_programs: Box<[TensorMapSharedRunProgram]>,
    payload_len: usize,
}

#[derive(Debug)]
struct TensorMapBoundGlobalProgram {
    runs: Vec<TensorMapByteRun>,
}

impl TensorMapTransferTemplate {
    fn compile(
        traversal_shape: &[usize],
        element_strides: &[usize],
        global_strides: &[usize],
        element_bits: usize,
        fp4_shared_layout: Option<Fp4SharedLayout>,
        swizzle_bytes: Option<usize>,
        swizzle_atomicity: SwizzleAtomicity,
    ) -> Result<Self, EngineError> {
        let geometry =
            tensor_map_geometry_from_metadata(traversal_shape, element_bits, fp4_shared_layout)?;
        let unit_count = geometry
            .outer_count
            .checked_mul(geometry.inner_units)
            .ok_or_else(|| EngineError::message("TensorMap transfer unit count overflow"))?;
        let payload_len = unit_count
            .checked_mul(geometry.unit_bytes)
            .ok_or_else(|| EngineError::message("TensorMap transfer payload size overflow"))?;
        let row_payload_len = geometry
            .inner_units
            .checked_mul(geometry.unit_bytes)
            .ok_or_else(|| EngineError::message("TensorMap row payload size overflow"))?;

        let max_coordinate_deltas = traversal_shape
            .iter()
            .zip(element_strides)
            .map(|(extent, stride)| {
                extent
                    .checked_sub(1)
                    .and_then(|value| value.checked_mul(*stride))
                    .and_then(|value| i64::try_from(value).ok())
                    .ok_or_else(|| EngineError::message("TensorMap coordinate delta overflow"))
            })
            .collect::<Result<Box<[_]>, _>>()?;

        let mut rows = Vec::with_capacity(geometry.outer_count);
        for outer_linear in 0..geometry.outer_count {
            let mut linear = outer_linear;
            let mut coordinate_deltas = Vec::with_capacity(traversal_shape.len().saturating_sub(1));
            let mut global_outer_byte_delta = 0_usize;
            for axis in 1..traversal_shape.len() {
                let coordinate = linear % traversal_shape[axis];
                linear /= traversal_shape[axis];
                let delta = coordinate
                    .checked_mul(element_strides[axis])
                    .ok_or_else(|| EngineError::message("TensorMap coordinate delta overflow"))?;
                global_outer_byte_delta =
                    global_outer_byte_delta
                        .checked_add(delta.checked_mul(global_strides[axis - 1]).ok_or_else(
                            || EngineError::message("TensorMap stride offset overflow"),
                        )?)
                        .ok_or_else(|| EngineError::message("TensorMap byte offset overflow"))?;
                coordinate_deltas
                    .push(i64::try_from(delta).map_err(|_| {
                        EngineError::message("TensorMap coordinate delta overflow")
                    })?);
            }
            rows.push(TensorMapRowTemplate {
                coordinate_deltas: coordinate_deltas.into_boxed_slice(),
                global_outer_byte_delta,
                payload_offset: outer_linear
                    .checked_mul(row_payload_len)
                    .ok_or_else(|| EngineError::message("TensorMap payload offset overflow"))?,
            });
        }
        let mut source_row_order = (0..rows.len()).collect::<Vec<_>>();
        source_row_order.sort_unstable_by_key(|row| {
            (
                rows[*row].global_outer_byte_delta,
                rows[*row].payload_offset,
            )
        });

        let phase_count = swizzle_bytes.map_or(1, |bytes| {
            if bytes == 96 { 2 } else { bytes / swizzle_atomicity.bytes() }
        });
        let mut shared_programs = Vec::with_capacity(phase_count);
        for pointer_phase in 0..phase_count {
            let absolute_base = pointer_phase
                .checked_mul(128)
                .ok_or_else(|| EngineError::message("TensorMap shared pointer phase overflow"))?;
            let mut runs = Vec::<TensorMapByteRun>::new();
            let mut byte_extent = 0_usize;
            for outer_linear in 0..geometry.outer_count {
                for inner_unit in 0..geometry.inner_units {
                    let unit_index = outer_linear
                        .checked_mul(geometry.inner_units)
                        .and_then(|base| base.checked_add(inner_unit))
                        .ok_or_else(|| EngineError::message("TensorMap transfer index overflow"))?;
                    let payload_offset = unit_index
                        .checked_mul(geometry.unit_bytes)
                        .ok_or_else(|| EngineError::message("TensorMap payload offset overflow"))?;
                    // Interleave slices split at 16B atoms. Accepted 8B-flip
                    // layouts already have transfer units of at most 8B.
                    let atom_bytes = geometry.unit_bytes.min(16);
                    for atom in (0..geometry.unit_bytes).step_by(atom_bytes) {
                        let payload_offset = payload_offset + atom;
                        let byte_offset = tensor_map_shared_byte_offset_from_layout(
                            swizzle_bytes,
                            swizzle_atomicity,
                            outer_linear,
                            inner_unit * geometry.unit_stride_bytes + atom,
                            geometry.inner_row_bytes,
                            absolute_base,
                        )?;
                        byte_extent =
                            byte_extent.max(byte_offset.checked_add(atom_bytes).ok_or_else(
                                || EngineError::message("TensorMap shared extent overflow"),
                            )?);
                        if let Some(previous) = runs.last_mut() {
                            let bytes_contiguous =
                                previous.byte_offset.checked_add(previous.byte_len)
                                    == Some(byte_offset);
                            let payload_contiguous =
                                previous.payload_offset.checked_add(previous.byte_len)
                                    == Some(payload_offset);
                            if bytes_contiguous && payload_contiguous {
                                previous.byte_len =
                                    previous.byte_len.checked_add(atom_bytes).ok_or_else(|| {
                                        EngineError::message("TensorMap shared run size overflow")
                                    })?;
                                continue;
                            }
                        }
                        runs.push(TensorMapByteRun {
                            byte_offset,
                            payload_offset,
                            byte_len: atom_bytes,
                        });
                    }
                }
            }
            shared_programs.push(TensorMapSharedRunProgram {
                runs: Arc::from(runs),
                byte_extent,
            });
        }

        Ok(Self {
            geometry,
            rows: Arc::from(rows),
            source_row_order: Arc::from(source_row_order),
            max_coordinate_deltas,
            shared_programs: shared_programs.into_boxed_slice(),
            payload_len,
        })
    }

    fn shared_program(&self, absolute_base: usize) -> &TensorMapSharedRunProgram {
        let phase = if self.shared_programs.len() == 1 {
            0
        } else {
            (absolute_base / 128) % self.shared_programs.len()
        };
        &self.shared_programs[phase]
    }

    fn bind_global(
        &self,
        tensor_map: &RuntimeTensorMap,
        origin: &[i64],
    ) -> Result<TensorMapBoundGlobalProgram, EngineError> {
        if origin.len() != tensor_map.global_shape.len() {
            return Err(EngineError::message(format!(
                "TensorMap expected {} coordinates, got {}",
                tensor_map.global_shape.len(),
                origin.len()
            )));
        }
        tensor_map.validate_u6_origin(origin)?;
        for (&coordinate, &max_delta) in origin.iter().zip(&self.max_coordinate_deltas) {
            coordinate
                .checked_add(max_delta)
                .ok_or_else(|| EngineError::message("TensorMap coordinate overflow"))?;
        }

        let extent = tensor_map.traversal_shape[0] as i128;
        let origin_inner = i128::from(origin[0]);
        let global_extent = tensor_map.global_shape[0] as i128;
        let step = tensor_map.element_strides[0] as i128;
        let ceil_div = |value: i128| -(-value).div_euclid(step);
        let start = ceil_div(-origin_inner).clamp(0, extent);
        let end = ceil_div(global_extent - origin_inner).clamp(0, extent);
        let packed = self.geometry.packed_elements as i128;
        if start % packed != 0 || end % packed != 0 {
            return Err(EngineError::message(
                "TensorMap valid FP4 interval is not aligned to a packed transfer unit",
            ));
        }
        if end <= start {
            return Ok(TensorMapBoundGlobalProgram { runs: Vec::new() });
        }
        let transfer_bits = tensor_map.transfer_element_bits() as i128;
        let mut inner_runs = Vec::new();
        let first_unit = (start / packed) as usize;
        let end_unit = (end / packed) as usize;
        let units_per_run = if step == 1 { end_unit - first_unit } else { 1 };
        for unit in (first_unit..end_unit).step_by(units_per_run) {
            let coordinate = origin_inner + unit as i128 * packed * step;
            let byte_offset = usize::try_from(coordinate * transfer_bits / 8)
                .map_err(|_| EngineError::message("TensorMap inner byte offset overflow"))?;
            inner_runs.push(TensorMapByteRun {
                byte_offset,
                payload_offset: unit.checked_mul(self.geometry.unit_bytes).ok_or_else(|| {
                    EngineError::message("TensorMap inner payload offset overflow")
                })?,
                byte_len: units_per_run
                    .checked_mul(self.geometry.unit_bytes)
                    .ok_or_else(|| EngineError::message("TensorMap inner run size overflow"))?,
            });
        }
        let mut origin_outer_byte_offset = 0_i128;
        for axis in 1..origin.len() {
            origin_outer_byte_offset = origin_outer_byte_offset
                .checked_add(
                    i128::from(origin[axis])
                        .checked_mul(tensor_map.global_strides[axis - 1] as i128)
                        .ok_or_else(|| EngineError::message("TensorMap stride offset overflow"))?,
                )
                .ok_or_else(|| EngineError::message("TensorMap byte offset overflow"))?;
        }

        let mut runs = Vec::<TensorMapByteRun>::new();
        for &row_index in self.source_row_order.iter() {
            let row = &self.rows[row_index];
            let mut in_bounds = true;
            for axis in 1..origin.len() {
                let coordinate = origin[axis]
                    .checked_add(row.coordinate_deltas[axis - 1])
                    .ok_or_else(|| EngineError::message("TensorMap coordinate overflow"))?;
                if coordinate < 0
                    || usize::try_from(coordinate)
                        .map_or(true, |value| value >= tensor_map.global_shape[axis])
                {
                    in_bounds = false;
                    break;
                }
            }
            if !in_bounds {
                continue;
            }
            for inner in &inner_runs {
                let run_byte_len = inner.byte_len;
                let byte_offset = usize::try_from(
                    origin_outer_byte_offset
                        .checked_add(row.global_outer_byte_delta as i128)
                        .and_then(|value| value.checked_add(inner.byte_offset as i128))
                        .ok_or_else(|| EngineError::message("TensorMap byte offset overflow"))?,
                )
                .map_err(|_| EngineError::message("TensorMap byte offset overflow"))?;
                let payload_offset = row
                    .payload_offset
                    .checked_add(inner.payload_offset)
                    .ok_or_else(|| EngineError::message("TensorMap payload offset overflow"))?;
                if let Some(previous) = runs.last_mut() {
                    let source_contiguous =
                        previous.byte_offset.checked_add(previous.byte_len) == Some(byte_offset);
                    let payload_contiguous = previous.payload_offset.checked_add(previous.byte_len)
                        == Some(payload_offset);
                    if source_contiguous && payload_contiguous {
                        previous.byte_len =
                            previous.byte_len.checked_add(run_byte_len).ok_or_else(|| {
                                EngineError::message("TensorMap source run size overflow")
                            })?;
                        continue;
                    }
                }
                runs.push(TensorMapByteRun {
                    byte_offset,
                    payload_offset,
                    byte_len: run_byte_len,
                });
            }
        }
        Ok(TensorMapBoundGlobalProgram { runs })
    }
}

fn tensor_map_transaction_bytes(unit_bytes: usize) -> Result<u64, EngineError> {
    u64::try_from(unit_bytes)
        .map_err(|_| EngineError::message("TensorMap transaction byte count overflow"))
}

fn raw_tma_shared_base(
    pointer: &PhysicalPtr,
    context: &WarpContext,
) -> Result<(usize, RuntimeBuffer, usize, usize), EngineError> {
    let mask = context.active_mask();
    let lane = mask
        .first_active()
        .ok_or_else(|| EngineError::message("raw TMA shared pointer has no active issuing lane"))?;
    let relative = pointer.lane_address_byte_offset(lane, 0)?;
    // One observer owns captured and target-encoded addresses. The rank byte
    // selects the CTA; only the byte-offset field participates in swizzling.
    let state = pointer.shared_byte_addresses_u32(
        context, WarpMask::from_bits(1 << lane),
    )?[lane];
    let absolute = crate::instruction_codec::shared_address_byte_offset(state) as usize;
    Ok((lane, pointer.buffer().clone(), relative, absolute))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawTmaG2cResult {
    bytes_per_target: u64,
    target_ctas: Vec<usize>,
}

impl RawTmaG2cResult {
    pub fn bytes_per_target(&self) -> u64 {
        self.bytes_per_target
    }

    pub fn target_ctas(&self) -> &[usize] {
        &self.target_ctas
    }

}

struct RawTmaG2cLayout {
    geometry: TensorMapGeometry,
    lane: usize,
    destination_buffer: RuntimeBuffer,
    base_absolute: usize,
    target_ctas: Vec<usize>,
    multicast: bool,
}

fn raw_tma_g2c_layout(
    context: &WarpContext,
    destination: &PhysicalPtr,
    tensor_map: &RuntimeTensorMap,
    origin: &[i64],
    cta_mask: u64,
    multicast: bool,
) -> Result<RawTmaG2cLayout, EngineError> {
    let geometry = tensor_map_geometry(tensor_map)?;
    if tensor_map.transfer_element_bits() == 4 {
        let required_alignment = match tensor_map.fp4_shared_layout {
            Some(Fp4SharedLayout::Align8Packed) => 2_i64,
            Some(Fp4SharedLayout::Align16Padded) => 128_i64,
            None => {
                return Err(EngineError::message(
                    "FP4 TensorMap is missing its shared layout",
                ));
            }
        };
        if origin
            .first()
            .is_some_and(|value| value.rem_euclid(required_alignment) != 0)
        {
            return Err(EngineError::message(format!(
                "FP4 TensorMap inner origin must be a multiple of {required_alignment}"
            )));
        }
    }
    let (lane, destination_buffer, _base_relative, base_absolute) =
        raw_tma_shared_base(destination, context)?;
    tensor_map.validate_swizzle_direction(true)?;
    let target_ctas = if multicast {
        let ctas = context.topology().ctas_per_cluster();
        super::io::validate_tma_multicast_mask(cta_mask, ctas)?;
        if !matches!(destination_buffer, RuntimeBuffer::Shared { .. }) {
            return Err(EngineError::message(
                "raw TMA multicast destination must be local shared memory",
            ));
        }
        (0..ctas)
            .filter(|target| cta_mask & (1_u64 << target) != 0)
            .collect::<Vec<_>>()
    } else {
        vec![context.cta_id_in_cluster()]
    };
    Ok(RawTmaG2cLayout {
        geometry,
        lane,
        destination_buffer,
        base_absolute,
        target_ctas,
        multicast,
    })
}

/// Push the accesses of one side of a transfer whose units are `spans`: one
/// unmerged batch carrying every unit, so the analysis sees exactly the
/// per-unit accesses it would see from one batch per unit without the
/// per-batch overhead. Units that overlap cannot share a footprint and keep
/// one batch each.
fn push_transfer_unit_batches(
    accesses: &mut Vec<PhysicalAccessBatch>,
    operation: &OperationContext,
    lane: usize,
    kind: PhysicalAccessKind,
    space: crate::PhysicalAccessSpace,
    spans: Vec<PhysicalByteSpan>,
    domain: ProxyMemoryDomain,
    siblings: usize,
    semantics: MemoryAccessSemantics,
) -> Result<(), EngineError> {
    if spans.is_empty() {
        return Ok(());
    }
    if transfer_units_overlap(&spans) || siblings > usize::from(u8::MAX) {
        for span in spans {
            accesses.push(
                single_lane_physical_access_batch(operation, lane, kind, space, vec![span])?
                    .with_proxy_memory_domain(domain)
                    .with_memory_semantics(semantics),
            );
        }
        return Ok(());
    }
    let mut batch =
        single_lane_physical_access_batch_unmerged(operation, lane, kind, space, spans)?
            .with_proxy_memory_domain(domain)
            .with_memory_semantics(semantics);
    if siblings > 1 {
        batch = batch.with_transfer_siblings(siblings);
    }
    accesses.push(batch);
    Ok(())
}

fn transfer_units_overlap(spans: &[PhysicalByteSpan]) -> bool {
    if spans.len() < 2 {
        return false;
    }
    let overlapping = |sorted: &[PhysicalByteSpan]| {
        sorted.windows(2).any(|pair| {
            pair[0].allocation() == pair[1].allocation()
                && pair[1].byte_offset() < pair[0].byte_end()
        })
    };
    if spans.is_sorted() {
        return overlapping(spans);
    }
    let mut sorted = spans.to_vec();
    sorted.sort_unstable();
    overlapping(&sorted)
}

fn canonicalize_physical_spans(
    mut spans: Vec<PhysicalByteSpan>,
) -> Result<Vec<PhysicalByteSpan>, EngineError> {
    spans.sort_unstable();
    let mut canonical = Vec::<PhysicalByteSpan>::with_capacity(spans.len());
    for span in spans {
        let Some(previous) = canonical.last_mut() else {
            canonical.push(span);
            continue;
        };
        if previous.allocation() != span.allocation() || previous.byte_end() < span.byte_offset() {
            canonical.push(span);
            continue;
        }
        let byte_end = previous.byte_end().max(span.byte_end());
        *previous = PhysicalByteSpan::new(
            previous.allocation(),
            previous.byte_offset(),
            byte_end - previous.byte_offset(),
        )
        .map_err(|error| EngineError::message(error.to_string()))?;
    }
    Ok(canonical)
}

fn bind_tensor_map_shared_runs(
    pointer: &PhysicalPtr,
    lane: usize,
    program: &TensorMapSharedRunProgram,
    write: bool,
) -> Result<TensorMapBoundSharedProgram, EngineError> {
    let base = if write {
        pointer.lane_write_byte_offset_at(lane, 0, program.byte_extent)?
    } else {
        pointer.lane_read_byte_offset_at(lane, 0, program.byte_extent)?
    };
    let runs = program
        .runs
        .iter()
        .map(|run| {
            Ok::<TensorMapByteRun, EngineError>(TensorMapByteRun {
                byte_offset: base
                    .checked_add(run.byte_offset)
                    .ok_or_else(|| EngineError::message("TensorMap shared offset overflow"))?,
                payload_offset: run.payload_offset,
                byte_len: run.byte_len,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(TensorMapBoundSharedProgram { runs })
}

/// TMA's TF32 conversion canonicalizes input NaNs (SM100 GPU bit oracle).
/// OOB fill is generated separately and must not pass through conversion.
pub(super) fn tma_f32_to_tf32(value: f32) -> f32 {
    if value.is_nan() {
        f32::from_bits(0x7fff_e000)
    } else {
        f32_to_tf32(value)
    }
}

fn materialize_raw_tma_g2c_payload(
    physical: &PhysicalMemory,
    tensor_map: &RuntimeTensorMap,
    source_runs: &[TensorMapByteRun],
    payload_len: usize,
    unit_bytes: usize,
    fill_mode: TensorMapFillMode,
    report_pattern: u32,
) -> Result<(Vec<u8>, bool), EngineError> {
    let mut payload = vec![0_u8; payload_len];
    if fill_mode == TensorMapFillMode::OobNan {
        if unit_bytes < 2 || !unit_bytes.is_multiple_of(2) {
            return Err(EngineError::message(
                "TensorMap OOB-NaN fill requires an even floating-point element width",
            ));
        }
        for unit in payload.chunks_exact_mut(unit_bytes) {
            for chunk in unit.chunks_exact_mut(2) {
                chunk.copy_from_slice(&crate::scalar::PTX_OOB_NAN.to_le_bytes());
            }
        }
    }
    for run in source_runs {
        tensor_map.read_global_bytes(
            physical,
            run.byte_offset,
            &mut payload[run.payload_offset..run.payload_offset + run.byte_len],
        )?;
    }
    let reported = super::memory_ops::copy_report_matches_runs(
        report_pattern,
        source_runs.iter().map(|run| {
            (
                run.byte_offset,
                &payload[run.payload_offset..run.payload_offset + run.byte_len],
            )
        }),
    )?;
    let round_tf32 = matches!(
        tensor_map.element_type,
        TensorMapElementType::Tf32 | TensorMapElementType::Tf32Ftz
    );
    // FTZ descriptor types affect tensor reductions, not copies. Convert only
    // in-bounds TF32 source data; hardware leaves OOB-NaN fill bits untouched.
    if round_tf32 {
        for run in source_runs {
            for element in
                payload[run.payload_offset..run.payload_offset + run.byte_len].chunks_exact_mut(4)
            {
                let raw: [u8; 4] = element.try_into().expect("four-byte TMA float element");
                element.copy_from_slice(&tma_f32_to_tf32(f32::from_le_bytes(raw)).to_le_bytes());
            }
        }
    }
    Ok((payload, reported))
}

/// Engine-private occurrence plan shared by numerical TMA execution and its
/// checker-visible physical footprint. It never crosses the generated ABI.
pub(crate) struct RawTmaG2cTransferPlan {
    layout: RawTmaG2cLayout,
    source_runs: Vec<TensorMapByteRun>,
    destination_runs: Vec<TensorMapByteRun>,
    payload_len: usize,
    fill_mode: TensorMapFillMode,
}

impl RawTmaG2cTransferPlan {
    pub(crate) fn im2col(
        context: &WarpContext,
        destination: &PhysicalPtr,
        tensor_map: &RuntimeTensorMap,
        origin: &[i64],
        mode: Im2colMode,
        info: &[i64],
        cta_mask: u64,
        multicast: bool,
    ) -> Result<Self, EngineError> {
        let origins = tensor_map.im2col_origins(origin, mode, info)?;
        let mut layout = raw_tma_g2c_layout(
            context,
            destination,
            tensor_map,
            origin,
            cta_mask,
            multicast,
        )?;
        layout.geometry.outer_count = origins.len();
        let template = &tensor_map.transfer_template;
        let (source_runs, destination_runs) = im2col_transfer_runs(
            tensor_map,
            &origins,
            destination,
            layout.lane,
            layout.base_absolute,
            true,
        )?;
        Ok(Self {
            layout,
            source_runs,
            destination_runs,
            payload_len: origins.len() * template.payload_len,
            fill_mode: tensor_map.fill_mode,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        context: &WarpContext,
        destination: &PhysicalPtr,
        tensor_map: &RuntimeTensorMap,
        origin: &[i64],
        cta_mask: u64,
        multicast: bool,
    ) -> Result<Self, EngineError> {
        if tensor_map.im2col.is_some() {
            return Err(EngineError::message(
                "tiled load requires a tiled TensorMap",
            ));
        }
        let layout = raw_tma_g2c_layout(
            context,
            destination,
            tensor_map,
            origin,
            cta_mask,
            multicast,
        )?;
        let template = &tensor_map.transfer_template;
        let source_program = template.bind_global(tensor_map, origin)?;
        let destination_program = bind_tensor_map_shared_runs(
            destination,
            layout.lane,
            template.shared_program(layout.base_absolute),
            true,
        )?;

        Ok(Self {
            layout,
            source_runs: source_program.runs,
            destination_runs: destination_program.runs,
            payload_len: template.payload_len,
            fill_mode: tensor_map.fill_mode,
        })
    }

    pub(crate) fn accesses_with(
        &self,
        operation: &OperationContext,
        context: &WarpContext,
        tensor_map: &RuntimeTensorMap,
        source: TmaSourceAccessPlan,
    ) -> Result<(Vec<PhysicalAccessBatch>, u64), EngineError> {
        let unit_bytes = self.layout.geometry.unit_bytes.min(16);
        let source_count = self.source_runs.iter().try_fold(0_usize, |count, run| {
            if !run.byte_len.is_multiple_of(unit_bytes) {
                return Err(EngineError::message(
                    "TensorMap global run is not transfer-unit aligned",
                ));
            }
            count
                .checked_add(run.byte_len / unit_bytes)
                .ok_or_else(|| EngineError::message("TensorMap source access count overflow"))
        })?;
        let destination_count = self
            .layout
            .geometry
            .outer_count
            .checked_mul(self.layout.geometry.inner_units)
            .and_then(|count| count.checked_mul(self.layout.target_ctas.len()))
            .ok_or_else(|| EngineError::message("TensorMap destination access count overflow"))?;
        let mut accesses = Vec::with_capacity(
            source_count
                .checked_add(destination_count)
                .ok_or_else(|| EngineError::message("TensorMap access count overflow"))?,
        );
        // Every transfer unit stays its own span (the footprints are built
        // unmerged), but all units of one side of the transfer travel in one
        // batch: the analysis sees exactly the per-unit accesses it would see
        // from one batch per unit, without paying per-batch overhead 400
        // times per transfer.
        let global_buffer = RuntimeBuffer::Global(tensor_map.view.clone());
        let mut source_space = None;
        let mut source_run_spans = Vec::with_capacity(self.source_runs.len());
        for run in &self.source_runs {
            // The source is one global view, whose resolution is affine in
            // the byte offset: resolving the whole run once bounds-checks
            // every unit, and the unit spans follow arithmetically.
            let access = resolve_runtime_physical_access(
                context,
                &global_buffer,
                self.layout.lane,
                run.byte_offset,
                run.byte_len,
                PhysicalAccessKind::Read,
            )?;
            if source_space.is_some_and(|space| space != access.space()) {
                return Err(EngineError::message(
                    "TensorMap source units resolved to different spaces",
                ));
            }
            source_space = Some(access.space());
            source_run_spans.push(access.span());
        }
        // Runs of a tensor-map traversal can overlap (a stride shorter than
        // the box row); the run form needs disjoint runs, so those fall back
        // to per-unit spans.
        let source_as_runs = source == TmaSourceAccessPlan::Runs
            && u32::try_from(unit_bytes).is_ok()
            && !transfer_units_overlap(&source_run_spans);
        if let Some(space) = source_space {
            if source_as_runs {
                accesses.push(
                    single_lane_transfer_run_batch(
                        operation,
                        self.layout.lane,
                        PhysicalAccessKind::Read,
                        space,
                        source_run_spans,
                        unit_bytes as u32,
                    )?
                    .with_proxy_memory_domain(ProxyMemoryDomain::Global),
                );
            } else {
                let mut source_spans = Vec::with_capacity(source_count);
                for run_span in &source_run_spans {
                    for unit_offset in (0..run_span.byte_len()).step_by(unit_bytes) {
                        source_spans.push(
                            PhysicalByteSpan::new(
                                run_span.allocation(),
                                run_span.byte_offset() + unit_offset,
                                unit_bytes,
                            )
                            .map_err(|_| {
                                EngineError::message("TensorMap global unit span overflow")
                            })?,
                        );
                    }
                }
                push_transfer_unit_batches(
                    &mut accesses,
                    operation,
                    self.layout.lane,
                    PhysicalAccessKind::Read,
                    space,
                    source_spans,
                    ProxyMemoryDomain::Global,
                    1,
                    MemoryAccessSemantics::plain(),
                )?;
            }
        }
        for target in self.layout.target_ctas.iter().copied() {
            let mut destination_space = None;
            let mut destination_spans =
                Vec::with_capacity(destination_count / self.layout.target_ctas.len().max(1));
            for run in &self.destination_runs {
                if !run.byte_len.is_multiple_of(unit_bytes) {
                    return Err(EngineError::message(
                        "TensorMap shared run is not transfer-unit aligned",
                    ));
                }
                for unit_offset in (0..run.byte_len).step_by(unit_bytes) {
                    let byte_offset = run
                        .byte_offset
                        .checked_add(unit_offset)
                        .ok_or_else(|| EngineError::message("TensorMap shared offset overflow"))?;
                    let access = if self.layout.multicast {
                        resolve_shared_runtime_physical_access_to_cta(
                            context,
                            &self.layout.destination_buffer,
                            self.layout.lane,
                            target,
                            byte_offset,
                            unit_bytes,
                            PhysicalAccessKind::Write,
                        )?
                    } else {
                        resolve_runtime_physical_access(
                            context,
                            &self.layout.destination_buffer,
                            self.layout.lane,
                            byte_offset,
                            unit_bytes,
                            PhysicalAccessKind::Write,
                        )?
                    };
                    if destination_space.is_some_and(|space| space != access.space()) {
                        return Err(EngineError::message(
                            "TensorMap destination units resolved to different spaces",
                        ));
                    }
                    destination_space = Some(access.space());
                    destination_spans.push(access.span());
                }
            }
            if let Some(space) = destination_space {
                push_transfer_unit_batches(
                    &mut accesses,
                    operation,
                    self.layout.lane,
                    PhysicalAccessKind::Write,
                    space,
                    destination_spans,
                    ProxyMemoryDomain::SharedCluster,
                    self.layout.target_ctas.len(),
                    MemoryAccessSemantics::plain(),
                )?;
            }
        }
        Ok((accesses, self.bytes_per_target()?))
    }

    pub(crate) fn execute_with_report(
        &self,
        physical: &PhysicalMemory,
        context: &WarpContext,
        tensor_map: &RuntimeTensorMap,
        report_pattern: u32,
    ) -> Result<(RawTmaG2cResult, bool), EngineError> {
        let (payload, reported) = materialize_raw_tma_g2c_payload(
            physical,
            tensor_map,
            &self.source_runs,
            self.payload_len,
            self.layout.geometry.unit_bytes,
            self.fill_mode,
            report_pattern,
        )?;
        if self.layout.multicast {
            for target in self.layout.target_ctas.iter().copied() {
                super::with_shared_runtime_write_session(
                    physical,
                    context,
                    &self.layout.destination_buffer,
                    self.layout.lane,
                    Some(target),
                    |writer| {
                        for run in &self.destination_runs {
                            writer.write_bytes_prevalidated(
                                run.byte_offset,
                                &payload[run.payload_offset..run.payload_offset + run.byte_len],
                            );
                        }
                        Ok(())
                    },
                )?;
            }
        } else {
            super::with_shared_runtime_write_session(
                physical,
                context,
                &self.layout.destination_buffer,
                self.layout.lane,
                None,
                |writer| {
                    for run in &self.destination_runs {
                        writer.write_bytes_prevalidated(
                            run.byte_offset,
                            &payload[run.payload_offset..run.payload_offset + run.byte_len],
                        );
                    }
                    Ok(())
                },
            )?;
        }
        Ok((
            RawTmaG2cResult {
                bytes_per_target: self.bytes_per_target()?,
                target_ctas: self.layout.target_ctas.clone(),
            },
            reported,
        ))
    }

    fn bytes_per_target(&self) -> Result<u64, EngineError> {
        u64::try_from(self.payload_len)
            .map_err(|_| EngineError::message("raw TMA delivered-byte count overflow"))
    }
}

/// Both directions use the same pixel walk, channel packing and swizzle.
/// Only the shared pointer's read/write permission differs.
fn im2col_transfer_runs(
    tensor_map: &RuntimeTensorMap,
    origins: &[Vec<i64>],
    shared: &PhysicalPtr,
    lane: usize,
    absolute_base: usize,
    write_shared: bool,
) -> Result<(Vec<TensorMapByteRun>, Vec<TensorMapByteRun>), EngineError> {
    let template = &tensor_map.transfer_template;
    let geometry = template.geometry;
    let mut global_runs = Vec::new();
    let mut shared_runs = Vec::new();
    for (pixel, origin) in origins.iter().enumerate() {
        let payload_base = pixel * template.payload_len;
        if write_shared || !matches!(tensor_map.transfer_element_bits(), 4 | 6) {
            for mut run in template.bind_global(tensor_map, origin)?.runs {
                run.payload_offset += payload_base;
                global_runs.push(run);
            }
        }
        for unit in 0..geometry.inner_units {
            // A 32B channel slice crosses two independently swizzled atoms.
            let atom_bytes = geometry.unit_bytes.min(16);
            for atom in (0..geometry.unit_bytes).step_by(atom_bytes) {
                let offset = tensor_map_shared_byte_offset(
                    tensor_map,
                    pixel,
                    unit * geometry.unit_stride_bytes + atom,
                    geometry.inner_row_bytes,
                    absolute_base,
                )?;
                let byte_offset = if write_shared {
                    shared.lane_write_byte_offset_at(lane, offset, atom_bytes)?
                } else {
                    shared.lane_read_byte_offset_at(lane, offset, atom_bytes)?
                };
                shared_runs.push(TensorMapByteRun {
                    byte_offset,
                    payload_offset: payload_base + unit * geometry.unit_bytes + atom,
                    byte_len: atom_bytes,
                });
            }
        }
    }
    Ok((global_runs, shared_runs))
}

#[derive(Clone, Copy, Debug)]
struct TensorMapS2gBitFragment {
    byte_offset: usize,
    payload_offset: usize,
    source_shift: usize,
    target_shift: usize,
    mask: u8,
}

fn append_s2g_bits(
    tensor_map: &RuntimeTensorMap,
    coordinates: &[i64],
    payload_offset: usize,
    fragments: &mut Vec<TensorMapS2gBitFragment>,
) -> Result<(), EngineError> {
    let mut coordinates = coordinates.to_vec();
    let bits = tensor_map.element_bits;
    let packed_elements = tensor_map.transfer_template.geometry.packed_elements;
    for packed_index in 0..packed_elements {
        if packed_index != 0 {
            coordinates[0] = coordinates[0]
                .checked_add(1)
                .ok_or_else(|| EngineError::message("sub-byte TensorMap coordinate overflow"))?;
        }
        if !tensor_map_coordinates_in_bounds(tensor_map, &coordinates) {
            continue;
        }
        let (byte_offset, target_shift) = tensor_map_global_byte_offset(tensor_map, &coordinates)?;
        // U4 input is nibble-packed; U6 input has one value per shared byte.
        let source_bit = packed_index * if bits == 6 { 8 } else { bits };
        let mut consumed = 0;
        while consumed < bits {
            let target_bit = target_shift + consumed;
            let width = (bits - consumed).min(8 - target_bit % 8);
            fragments.push(TensorMapS2gBitFragment {
                byte_offset: byte_offset + target_bit / 8,
                payload_offset: payload_offset + source_bit / 8,
                source_shift: source_bit % 8 + consumed,
                target_shift: target_bit % 8,
                mask: ((1_u16 << width) - 1) as u8,
            });
            consumed += width;
        }
    }
    Ok(())
}

/// The U6 load template names twelve payload bytes per sixteen-byte atom.
/// Stores instead read all sixteen bytes (one low-six-bit value in each).
fn expand_u6_source_runs(
    runs: &mut [TensorMapByteRun],
    payload_len: usize,
) -> Result<usize, EngineError> {
    let expand = |bytes: usize| {
        debug_assert!(bytes.is_multiple_of(12));
        (bytes / 12)
            .checked_mul(16)
            .ok_or_else(|| EngineError::message("U6 TensorMap source payload overflow"))
    };
    for run in runs {
        debug_assert_eq!(run.byte_len, 12);
        run.payload_offset = expand(run.payload_offset)?;
        run.byte_len = 16;
    }
    expand(payload_len)
}

/// Engine-private shared-to-global counterpart of
/// [`RawTmaG2cTransferPlan`]. The physical footprint and numerical snapshot
/// consume the same resolved source runs and destination fragments.
pub(crate) struct RawTmaS2gTransferPlan {
    lane: usize,
    source_buffer: RuntimeBuffer,
    source_runs: Vec<TensorMapByteRun>,
    destination_runs: Vec<TensorMapByteRun>,
    destination_bits: Vec<TensorMapS2gBitFragment>,
    unit_bytes: usize,
    payload_len: usize,
}

impl RawTmaS2gTransferPlan {
    pub(crate) fn im2col(
        context: &WarpContext,
        source: &PhysicalPtr,
        tensor_map: &RuntimeTensorMap,
        origin: &[i64],
        mode: Im2colMode,
    ) -> Result<Self, EngineError> {
        let bounds = tensor_map
            .im2col
            .as_ref()
            .ok_or_else(|| EngineError::message("im2col store requires an im2col TensorMap"))?;
        if origin.iter().any(|value| *value < 0)
            || bounds.lower.iter().any(|value| *value < 0)
            || bounds.upper.iter().any(|value| *value > 0)
        {
            return Err(EngineError::message(
                "im2col store requires nonnegative coordinates and bounds inside the tensor",
            ));
        }
        if tensor_map.fp4_shared_layout == Some(Fp4SharedLayout::Align16Padded) {
            return Err(EngineError::message(
                "align16 padded FP4 TensorMap does not support shared-to-global Tensor Copy",
            ));
        }
        let count = if mode == Im2colMode::Spatial {
            tensor_map.rank() - 2
        } else {
            2
        };
        let mut origins = tensor_map.im2col_origins(origin, mode, &vec![0; count])?;
        if mode == Im2colMode::Wide {
            // Once a pixel leaves the tensor, no subsequent pixel is read
            // from shared memory or written/reduced to global memory.
            let valid = origins
                .iter()
                .take_while(|coordinates| tensor_map_coordinates_in_bounds(tensor_map, coordinates))
                .count();
            origins.truncate(valid);
        }
        let (lane, source_buffer, _, absolute_base) = raw_tma_shared_base(source, context)?;
        tensor_map.validate_swizzle_direction(false)?;
        let (destination_runs, mut source_runs) =
            im2col_transfer_runs(tensor_map, &origins, source, lane, absolute_base, false)?;
        let geometry = tensor_map.transfer_template.geometry;
        let mut pixel_bytes = tensor_map.transfer_template.payload_len;
        let unit_bytes = if tensor_map.element_bits == 6 {
            pixel_bytes = expand_u6_source_runs(&mut source_runs, pixel_bytes)?;
            16
        } else {
            geometry.unit_bytes.min(16)
        };
        let mut destination_bits = Vec::new();
        if matches!(tensor_map.transfer_element_bits(), 4 | 6) {
            for (pixel, origin) in origins.iter().enumerate() {
                tensor_map.validate_u6_origin(origin)?;
                for unit in 0..tensor_map.transfer_template.geometry.inner_units {
                    let mut coordinates = origin.clone();
                    coordinates[0] = coordinates[0]
                        .checked_add((unit * geometry.packed_elements) as i64)
                        .ok_or_else(|| EngineError::message("sub-byte TensorMap coordinate overflow"))?;
                    append_s2g_bits(
                        tensor_map,
                        &coordinates,
                        pixel * pixel_bytes + unit * unit_bytes,
                        &mut destination_bits,
                    )?;
                }
            }
        }
        Ok(Self {
            lane,
            source_buffer,
            source_runs,
            destination_runs,
            destination_bits,
            unit_bytes,
            payload_len: origins.len() * pixel_bytes,
        })
    }

    pub(crate) fn new(
        context: &WarpContext,
        source: &PhysicalPtr,
        tensor_map: &RuntimeTensorMap,
        origin: &[i64],
    ) -> Result<Self, EngineError> {
        // PTX tensor-copy direction restrictions: tiled shared-to-global
        // copies/reductions cannot start at negative tensor coordinates.
        if origin.iter().any(|coordinate| *coordinate < 0) {
            return Err(EngineError::message(
                "tiled TMA store requires nonnegative starting coordinates",
            ));
        }
        if tensor_map.im2col.is_some() {
            return Err(EngineError::message(
                "tiled store requires a tiled TensorMap",
            ));
        }
        if tensor_map.fp4_shared_layout == Some(Fp4SharedLayout::Align16Padded) {
            return Err(EngineError::message(
                "align16 padded FP4 TensorMap does not support shared-to-global Tensor Copy",
            ));
        }
        let template = &tensor_map.transfer_template;
        let geometry = template.geometry;
        let (lane, source_buffer, _base_relative, base_absolute) =
            raw_tma_shared_base(source, context)?;
        tensor_map.validate_swizzle_direction(false)?;
        let mut source_program = bind_tensor_map_shared_runs(
            source,
            lane,
            template.shared_program(base_absolute),
            false,
        )?;

        let mut payload_len = template.payload_len;
        let unit_bytes = if tensor_map.element_bits == 6 {
            payload_len = expand_u6_source_runs(&mut source_program.runs, payload_len)?;
            16
        } else {
            geometry.unit_bytes.min(16)
        };
        let mut destination_bits = Vec::new();
        let destination_runs = if matches!(tensor_map.transfer_element_bits(), 4 | 6) {
            if origin.len() != tensor_map.global_shape.len() {
                return Err(EngineError::message(format!(
                    "TensorMap expected {} coordinates, got {}",
                    tensor_map.global_shape.len(),
                    origin.len()
                )));
            }
            tensor_map.validate_u6_origin(origin)?;
            let mut coordinates = vec![0_i64; tensor_map.global_shape.len()];
            for outer_linear in 0..geometry.outer_count {
                let outer = tensor_map_outer_coordinates(tensor_map, outer_linear);
                for inner_unit in 0..geometry.inner_units {
                    let payload_offset = outer_linear
                        .checked_mul(geometry.inner_units)
                        .and_then(|base| base.checked_add(inner_unit))
                        .and_then(|unit| unit.checked_mul(unit_bytes))
                        .ok_or_else(|| EngineError::message("TensorMap payload offset overflow"))?;
                    let inner_element = inner_unit * geometry.packed_elements;
                    tensor_map_global_coordinates_into(
                        tensor_map,
                        origin,
                        inner_element,
                        &outer,
                        &mut coordinates,
                    )?;
                    append_s2g_bits(
                        tensor_map,
                        &coordinates,
                        payload_offset,
                        &mut destination_bits,
                    )?;
                }
            }
            Vec::new()
        } else {
            let program = template.bind_global(tensor_map, origin)?;
            program.runs
        };

        Ok(Self {
            lane,
            source_buffer,
            source_runs: source_program.runs,
            destination_runs,
            destination_bits,
            unit_bytes,
            payload_len,
        })
    }

    pub(crate) fn accesses(
        &self,
        operation: &OperationContext,
        context: &WarpContext,
        tensor_map: &RuntimeTensorMap,
        destination_kind: PhysicalAccessKind,
    ) -> Result<(Vec<PhysicalAccessBatch>, Vec<PhysicalAccessBatch>), EngineError> {
        if !matches!(
            destination_kind,
            PhysicalAccessKind::Write | PhysicalAccessKind::AtomicReadModifyWrite
        ) {
            return Err(EngineError::message(
                "raw TMA shared-to-global destination must write or atomically reduce",
            ));
        }
        let destination_semantics = match destination_kind {
            PhysicalAccessKind::Write => MemoryAccessSemantics::plain(),
            PhysicalAccessKind::AtomicReadModifyWrite => MemoryAccessSemantics::async_reduction(),
            _ => unreachable!("destination kind validated above"),
        };
        let mut source_accesses = Vec::new();
        let mut source_space = None;
        let mut source_spans = Vec::new();
        for run in &self.source_runs {
            if !run.byte_len.is_multiple_of(self.unit_bytes) {
                return Err(EngineError::message(
                    "TensorMap shared run is not transfer-unit aligned",
                ));
            }
            for unit_offset in (0..run.byte_len).step_by(self.unit_bytes) {
                let byte_offset = run
                    .byte_offset
                    .checked_add(unit_offset)
                    .ok_or_else(|| EngineError::message("TensorMap shared offset overflow"))?;
                let access = resolve_runtime_physical_access(
                    context,
                    &self.source_buffer,
                    self.lane,
                    byte_offset,
                    self.unit_bytes,
                    PhysicalAccessKind::Read,
                )?;
                if source_space.is_some_and(|space| space != access.space()) {
                    return Err(EngineError::message(
                        "TensorMap source units resolved to different spaces",
                    ));
                }
                source_space = Some(access.space());
                source_spans.push(access.span());
            }
        }
        if let Some(space) = source_space {
            push_transfer_unit_batches(
                &mut source_accesses,
                operation,
                self.lane,
                PhysicalAccessKind::Read,
                space,
                source_spans,
                ProxyMemoryDomain::SharedCta,
                1,
                MemoryAccessSemantics::plain(),
            )?;
        }
        let mut destination_accesses = Vec::new();
        let global_buffer = RuntimeBuffer::Global(tensor_map.view.clone());
        let mut destination_space = None;
        let mut destination_spans = Vec::new();
        for run in &self.destination_runs {
            if !run.byte_len.is_multiple_of(self.unit_bytes) {
                return Err(EngineError::message(
                    "TensorMap global run is not transfer-unit aligned",
                ));
            }
            for unit_offset in (0..run.byte_len).step_by(self.unit_bytes) {
                let byte_offset = run
                    .byte_offset
                    .checked_add(unit_offset)
                    .ok_or_else(|| EngineError::message("TensorMap global offset overflow"))?;
                let access = resolve_runtime_physical_access(
                    context,
                    &global_buffer,
                    self.lane,
                    byte_offset,
                    self.unit_bytes,
                    destination_kind,
                )?;
                if destination_space.is_some_and(|space| space != access.space()) {
                    return Err(EngineError::message(
                        "TensorMap destination units resolved to different spaces",
                    ));
                }
                destination_space = Some(access.space());
                destination_spans.push(access.span());
            }
        }
        if let Some(space) = destination_space {
            push_transfer_unit_batches(
                &mut destination_accesses,
                operation,
                self.lane,
                destination_kind,
                space,
                destination_spans,
                ProxyMemoryDomain::Global,
                1,
                destination_semantics,
            )?;
        }
        let mut fragment_index = 0_usize;
        while fragment_index < self.destination_bits.len() {
            let payload_offset = self.destination_bits[fragment_index].payload_offset;
            let mut spans = Vec::new();
            while fragment_index < self.destination_bits.len()
                && self.destination_bits[fragment_index].payload_offset == payload_offset
            {
                let fragment = self.destination_bits[fragment_index];
                spans.push(
                    resolve_runtime_physical_access(
                        context,
                        &global_buffer,
                        self.lane,
                        fragment.byte_offset,
                        1,
                        destination_kind,
                    )?
                    .span(),
                );
                fragment_index += 1;
            }
            let spans = canonicalize_physical_spans(spans)?;
            if !spans.is_empty() {
                destination_accesses.push(
                    single_lane_physical_access_batch(
                        operation,
                        self.lane,
                        destination_kind,
                        crate::PhysicalAccessSpace::Global,
                        spans,
                    )?
                    .with_proxy_memory_domain(ProxyMemoryDomain::Global)
                    .with_memory_semantics(destination_semantics),
                );
            }
        }
        Ok((source_accesses, destination_accesses))
    }

    pub(crate) fn execute(
        &self,
        physical: &PhysicalMemory,
        context: &WarpContext,
        tensor_map: &RuntimeTensorMap,
        reduction: Option<RawTmaReductionOp>,
    ) -> Result<Vec<DeferredGlobalWrite>, EngineError> {
        let reduction = reduction
            .map(|operation| tensor_map.reduction(operation))
            .transpose()?;

        let mut payload = vec![0_u8; self.payload_len];
        for run in &self.source_runs {
            let bytes = read_runtime_bytes(
                physical,
                context,
                &self.source_buffer,
                self.lane,
                run.byte_offset,
                run.byte_len,
            )?;
            payload[run.payload_offset..run.payload_offset + run.byte_len].copy_from_slice(&bytes);
        }

        let mut writes = Vec::new();
        for run in &self.destination_runs {
            let bytes = &payload[run.payload_offset..run.payload_offset + run.byte_len];
            if let Some(operation) = reduction {
                let element_bytes = tensor_map.element_bits / 8;
                for element_offset in (0..run.byte_len).step_by(element_bytes) {
                    let element = &bytes[element_offset..element_offset + element_bytes];
                    writes.push(physical.global().defer_reduction_write_bytes(
                        &tensor_map.view,
                        run.byte_offset + element_offset,
                        element.to_vec(),
                        operation,
                    )?);
                }
            } else {
                physical.global().defer_or_extend_write_bytes(
                    &mut writes,
                    &tensor_map.view,
                    run.byte_offset,
                    bytes,
                )?;
            }
        }
        for fragment in &self.destination_bits {
            let mask = fragment.mask << fragment.target_shift;
            let source_bits =
                (payload[fragment.payload_offset] >> fragment.source_shift) & fragment.mask;
            writes.push(physical.global().defer_masked_write_bytes(
                &tensor_map.view,
                fragment.byte_offset,
                vec![source_bits << fragment.target_shift],
                vec![mask],
            )?);
        }
        Ok(writes)
    }
}

/// How a G2C plan reports the global source side of the transfer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TmaSourceAccessPlan {
    /// One span per transfer unit (element), the exact reporting form.
    Units,
    /// One span per contiguous run of units, carrying the unit size; the
    /// analysis derives unit identities from the run.
    Runs,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_mbarrier_completion_targets(
    context: &WarpContext,
    barrier_pointer: &PhysicalPtr,
    issue_mask: WarpMask,
    source_barrier: PhysicalBarrierId,
    cta_group: i64,
    cta_mask: u64,
    multicast: bool,
) -> Result<PhysicalMbarrierCompletionTargets, EngineError> {
    if !multicast {
        return Ok(PhysicalMbarrierCompletionTargets::single(source_barrier));
    }
    resolve_tma_multicast_completion_targets(
        context,
        barrier_pointer,
        issue_mask,
        source_barrier,
        cta_group,
        cta_mask,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_tma_multicast_completion_targets(
    context: &WarpContext,
    barrier_pointer: &PhysicalPtr,
    issue_mask: WarpMask,
    source_barrier: PhysicalBarrierId,
    cta_group: i64,
    cta_mask: u64,
) -> Result<PhysicalMbarrierCompletionTargets, EngineError> {
    let ctas_per_cluster = context.topology().ctas_per_cluster();
    super::io::validate_tma_multicast_mask(cta_mask, ctas_per_cluster)?;
    let target_ctas = (0..ctas_per_cluster)
        .filter(|target| cta_mask & (1_u64 << target) != 0)
        .collect::<Vec<_>>();
    resolve_tma_target_completion_targets(
        context,
        barrier_pointer,
        issue_mask,
        source_barrier,
        cta_group,
        &target_ctas,
    )
}

#[allow(clippy::too_many_arguments)]
fn resolve_tma_target_completion_targets(
    context: &WarpContext,
    barrier_pointer: &PhysicalPtr,
    issue_mask: WarpMask,
    source_barrier: PhysicalBarrierId,
    cta_group: i64,
    target_ctas: &[usize],
) -> Result<PhysicalMbarrierCompletionTargets, EngineError> {
    if !matches!(cta_group, 1 | 2) {
        return Err(EngineError::message(format!(
            "TMA multicast requires cta_group 1 or 2, got {cta_group}"
        )));
    }
    if target_ctas.is_empty() {
        return Err(EngineError::message(
            "TMA multicast requires at least one target CTA",
        ));
    }

    let ctas_per_cluster = context.topology().ctas_per_cluster();
    if cta_group == 2 && (ctas_per_cluster < 2 || !ctas_per_cluster.is_multiple_of(2)) {
        return Err(EngineError::message(format!(
            "TMA cta_group=2 requires complete CTA pairs, got cluster size {ctas_per_cluster}"
        )));
    }
    let cluster_base = context
        .cluster_id()
        .checked_mul(ctas_per_cluster)
        .ok_or_else(|| EngineError::message("TMA cluster base overflow"))?;
    let source_local = source_barrier
        .target_global_cta_id()
        .checked_sub(cluster_base)
        .filter(|target| *target < ctas_per_cluster)
        .ok_or_else(|| EngineError::message("TMA mbarrier is outside the issuing cluster"))?;
    let barrier_parity = source_local & 1;

    let mut barrier_ids = Vec::with_capacity(target_ctas.len());
    for target_cta in target_ctas.iter().copied() {
        if target_cta >= ctas_per_cluster {
            return Err(EngineError::message(format!(
                "TMA multicast target CTA {target_cta} is outside cluster size {ctas_per_cluster}"
            )));
        }
        let barrier_target = if cta_group == 1 {
            target_cta
        } else {
            let target = (target_cta & !1_usize)
                .checked_add(barrier_parity)
                .ok_or_else(|| EngineError::message("TMA CTA-pair target overflow"))?;
            if target >= ctas_per_cluster {
                return Err(EngineError::message(format!(
                    "TMA cta_group=2 target CTA {target_cta} has no complete CTA pair"
                )));
            }
            target
        };
        let barrier = barrier_pointer.resolve_shared_barrier_multicast(
            context,
            issue_mask,
            barrier_target,
        )?;
        barrier_ids.push(barrier);
    }
    Ok(PhysicalMbarrierCompletionTargets::from_barrier_ids(
        barrier_ids,
    ))
}

/// Resolved gather4 geometry shared by the footprint planner and the transfer.
struct RawTmaGather4Layout {
    geometry: TensorMapGeometry,
    lane: usize,
    destination_buffer: RuntimeBuffer,
    base_absolute: usize,
    gather_origins: Vec<[i64; 2]>,
    target_ctas: Vec<usize>,
    multicast: bool,
}

fn raw_tma_gather4_layout(
    context: &WarpContext,
    destination: &PhysicalPtr,
    tensor_map: &RuntimeTensorMap,
    column: i64,
    rows: &[i64],
    cta_mask: u64,
    multicast: bool,
) -> Result<RawTmaGather4Layout, EngineError> {
    if tensor_map.global_shape.len() != 2
        || tensor_map.box_shape.len() != 2
        || tensor_map.box_shape[1] != 1
    {
        return Err(EngineError::message(
            "TensorMap gather4 requires a rank-2 map with outer box extent one",
        ));
    }
    if rows.len() != 4 {
        return Err(EngineError::message(format!(
            "TensorMap gather4 requires exactly four rows, got {}",
            rows.len()
        )));
    }
    let geometry = tensor_map_geometry(tensor_map)?;
    if tensor_map.transfer_element_bits() == 4 {
        let required_alignment = match tensor_map.fp4_shared_layout {
            Some(Fp4SharedLayout::Align8Packed) => 2_i64,
            Some(Fp4SharedLayout::Align16Padded) => 128_i64,
            None => {
                return Err(EngineError::message(
                    "FP4 TensorMap is missing its shared layout",
                ));
            }
        };
        if column.rem_euclid(required_alignment) != 0 {
            return Err(EngineError::message(format!(
                "FP4 TensorMap gather4 inner origin must be a multiple of {required_alignment}"
            )));
        }
    }
    if geometry.outer_count != 1 {
        return Err(EngineError::message(
            "TensorMap gather4 source bounding box must contain one row",
        ));
    }
    let gather_origins = rows
        .iter()
        .copied()
        .map(|row| [column, row])
        .collect::<Vec<_>>();
    let (lane, destination_buffer, _base_relative, base_absolute) =
        raw_tma_shared_base(destination, context)?;
    tensor_map.validate_swizzle_direction(true)?;
    if !matches!(destination_buffer, RuntimeBuffer::Shared { .. }) {
        return Err(EngineError::message(
            "TensorMap gather4 destination must be local shared memory",
        ));
    }
    let target_ctas = if multicast {
        super::io::validate_tma_multicast_mask(cta_mask, context.topology().ctas_per_cluster())?;
        (0..context.topology().ctas_per_cluster())
            .filter(|target| cta_mask & (1_u64 << target) != 0)
            .collect::<Vec<_>>()
    } else {
        vec![context.cta_id_in_cluster()]
    };
    Ok(RawTmaGather4Layout {
        geometry,
        lane,
        destination_buffer,
        base_absolute,
        gather_origins,
        target_ctas,
        multicast,
    })
}

impl RawTmaG2cTransferPlan {
    /// Gather differs only in the four selected source rows. Keep payload,
    /// report sampling, access footprints and completion on the tiled plan.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gather4(
        context: &WarpContext,
        destination: &PhysicalPtr,
        tensor_map: &RuntimeTensorMap,
        column: i64,
        rows: &[i64],
        cta_mask: u64,
        multicast: bool,
    ) -> Result<Self, EngineError> {
        let gathered = raw_tma_gather4_layout(
            context,
            destination,
            tensor_map,
            column,
            rows,
            cta_mask,
            multicast,
        )?;
        let template = &tensor_map.transfer_template;
        let payload_len = template
            .payload_len
            .checked_mul(4)
            .ok_or_else(|| EngineError::message("gather4 payload size overflow"))?;
        let mut source_runs = Vec::new();
        let mut destination_runs = Vec::new();
        for (row, origin) in gathered.gather_origins.iter().enumerate() {
            let payload_base = row * template.payload_len;
            for mut run in template.bind_global(tensor_map, origin)?.runs {
                run.payload_offset += payload_base;
                source_runs.push(run);
            }
            for unit in 0..gathered.geometry.inner_units {
                let box_offset = tensor_map_shared_byte_offset(
                    tensor_map,
                    row,
                    unit * gathered.geometry.unit_stride_bytes,
                    gathered.geometry.inner_row_bytes,
                    gathered.base_absolute,
                )?;
                destination_runs.push(TensorMapByteRun {
                    byte_offset: destination.lane_write_byte_offset_at(
                        gathered.lane,
                        box_offset,
                        gathered.geometry.unit_bytes,
                    )?,
                    payload_offset: payload_base + unit * gathered.geometry.unit_bytes,
                    byte_len: gathered.geometry.unit_bytes,
                });
            }
        }
        let mut geometry = gathered.geometry;
        geometry.outer_count = 4;
        Ok(Self {
            layout: RawTmaG2cLayout {
                geometry,
                lane: gathered.lane,
                destination_buffer: gathered.destination_buffer,
                base_absolute: gathered.base_absolute,
                target_ctas: gathered.target_ctas,
                multicast: gathered.multicast,
            },
            source_runs,
            destination_runs,
            payload_len,
            fill_mode: tensor_map.fill_mode,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::runtime::write_runtime_bytes;
    use std::{collections::BTreeSet, sync::Arc};

    use crate::{
        CtaId, DynamicOpId, GlobalMemory, LaunchTopology, OperationKind, PhysicalAccessSpace,
        PhysicalAllocationId, StaticOpId, WarpValue,
    };

    use super::*;

    #[test]
    fn replacement_fields_use_ptx_encodings_and_preserve_other_image_fields() {
        let (_, map) = make_map(
            vec![0; 256],
            vec![8, 8],
            vec![32],
            vec![8, 8],
            32,
            None,
            None,
        );
        let original = RuntimeTensorMapImage::from_tensor_map(&map);
        for (field, limit) in [
            ("box_dim", MAX_BOX_DIMENSION),
            ("element_stride", MAX_ELEMENT_STRIDE),
        ] {
            for index in 0..5 {
                for value in [1, 2, limit] {
                    let mut image = original.clone();
                    image.replace_field(field, Some(index), value).unwrap();
                    let mut expected = original.clone();
                    if field == "box_dim" {
                        expected.box_shape[index] = value;
                    } else {
                        expected.element_strides[index] = value;
                    }
                    assert_eq!(
                        RuntimeTensorMapImage::decode(&image.encode().unwrap()).unwrap(),
                        expected
                    );
                }
            }
            for (index, value) in [(0, 0), (0, limit + 1), (0, usize::MAX), (5, 1)] {
                let mut image = original.clone();
                assert!(image.replace_field(field, Some(index), value).is_err());
                assert_eq!(image, original);
            }
        }
        for (code, dtype) in [
            (0, TensorMapElementType::U8),
            (1, TensorMapElementType::U16),
            (2, TensorMapElementType::U32),
            (3, TensorMapElementType::I32),
            (4, TensorMapElementType::U64),
            (5, TensorMapElementType::I64),
            (6, TensorMapElementType::F16),
            (7, TensorMapElementType::F32),
            (8, TensorMapElementType::F32Ftz),
            (9, TensorMapElementType::F64),
            (10, TensorMapElementType::Bf16),
            (11, TensorMapElementType::Tf32),
            (12, TensorMapElementType::Tf32Ftz),
        ] {
            let mut image = original.clone();
            image.replace_field("elemtype", None, code).unwrap();
            let expected = RuntimeTensorMapImage {
                element_type: dtype,
                ..original.clone()
            };
            assert_eq!(
                RuntimeTensorMapImage::decode(&image.encode().unwrap()).unwrap(),
                expected
            );
        }
        for (field, value) in [
            ("rank", 4),
            ("swizzle_mode", 3),
            ("swizzle_mode", 4),
            ("fill_mode", 1),
            ("interleave_layout", 0),
            ("interleave_layout", 1),
            ("interleave_layout", 2),
        ] {
            let mut image = original.clone();
            image.replace_field(field, None, value).unwrap();
            let mut expected = original.clone();
            match field {
                "rank" => expected.rank = 5,
                "swizzle_mode" => expected.swizzle_bytes = Some(if value == 4 { 96 } else { 128 }),
                "fill_mode" => expected.fill_mode = TensorMapFillMode::OobNan,
                "interleave_layout" => {
                    expected.interleave_bytes = match value {
                        0 => None,
                        1 => Some(16),
                        2 => Some(32),
                        _ => unreachable!(),
                    }
                }
                _ => unreachable!(),
            }
            assert_eq!(
                RuntimeTensorMapImage::decode(&image.encode().unwrap()).unwrap(),
                expected
            );
        }
        for (field, value) in [
            ("rank", 5),
            ("elemtype", 13),
            ("elemtype", 16),
            ("swizzle_mode", 5),
            ("fill_mode", 2),
            ("interleave_layout", 3),
        ] {
            let mut image = original.clone();
            assert!(image.replace_field(field, None, value).is_err());
            assert_eq!(image, original);
        }
    }

    #[test]
    fn private_image_round_trips_documented_field_maxima() {
        let maximum_stride = usize::try_from(MAX_GLOBAL_STRIDE - 16).unwrap();
        let maximum_dimension = usize::try_from(MAX_GLOBAL_DIMENSION).unwrap();
        let image = RuntimeTensorMapImage {
            allocation_id: u64::MAX,
            im2col: None,
            base_byte_offset: usize::MAX,
            host_address: true,
            rank: 5,
            physical_global_shape: [maximum_dimension; 5],
            physical_global_strides: [0, 16, maximum_stride, 32],
            box_shape: [MAX_BOX_DIMENSION; 5],
            element_strides: [MAX_ELEMENT_STRIDE; 5],
            element_type: TensorMapElementType::U32x2,
            interleave_bytes: None,
            fp4_shared_layout: None,
            swizzle_bytes: Some(128),
            swizzle_atomicity: SwizzleAtomicity::B16,
            fill_mode: TensorMapFillMode::Zero,
        };

        let encoded = image.encode().unwrap();
        assert_eq!(encoded.len(), TENSOR_MAP_PAYLOAD_BYTES);
        assert_eq!(u32::from_le_bytes(encoded[16..20].try_into().unwrap()), 0);
        assert_eq!(RuntimeTensorMapImage::decode(&encoded).unwrap(), image);
    }

    #[test]
    fn override_decodes_all_four_stride_nibbles_without_mutating_source() {
        let global = GlobalMemory::new();
        let allocation = global.allocate_zeroed(128 * 1024).unwrap();
        let view = global.full_view(allocation).unwrap();
        let original = RuntimeTensorMap::new(
            view.clone(),
            vec![4, 1, 1, 1, 1],
            vec![16; 4],
            vec![4, 1, 1, 1, 1],
            vec![1; 5],
            32,
            TensorMapElementType::F32,
            None,
            None,
            TensorMapFillMode::Zero,
        )
        .unwrap();
        let result = original
            .with_overrides(
                &global,
                view,
                &[4, 1, 1, 1, 1],
                &[1, 2, 3, 4],
                0x4321,
                &[0; 5],
            )
            .unwrap();
        for axis in 0..4 {
            assert_eq!(
                result.physical_global_strides[axis],
                (((axis + 1) << 32) | (axis + 1)) << 4
            );
        }
        assert_eq!(original.physical_global_strides, [16; 4]);
    }

    #[test]
    fn private_image_write_leaves_descriptor_tail_untouched() {
        let global = GlobalMemory::new();
        let allocation = global
            .allocate_from_bytes(vec![0xa5; TENSOR_MAP_DESCRIPTOR_BYTES])
            .unwrap();
        let descriptor = global.full_view(allocation).unwrap();
        let image = RuntimeTensorMapImage {
            allocation_id: 7,
            im2col: None,
            base_byte_offset: 16,
            host_address: false,
            rank: 1,
            physical_global_shape: [16, 1, 1, 1, 1],
            physical_global_strides: [0; 4],
            box_shape: [16, 1, 1, 1, 1],
            element_strides: [1; 5],
            element_type: TensorMapElementType::U8,
            interleave_bytes: None,
            fp4_shared_layout: None,
            swizzle_bytes: None,
            swizzle_atomicity: SwizzleAtomicity::B16,
            fill_mode: TensorMapFillMode::Zero,
        };

        image.write(&global, &descriptor).unwrap();

        assert_eq!(
            RuntimeTensorMapImage::read(&global, &descriptor).unwrap(),
            image
        );
        assert_eq!(
            global
                .read_bytes(
                    &descriptor,
                    TENSOR_MAP_PAYLOAD_BYTES,
                    TENSOR_MAP_DESCRIPTOR_BYTES - TENSOR_MAP_PAYLOAD_BYTES,
                )
                .unwrap(),
            vec![0xa5; TENSOR_MAP_DESCRIPTOR_BYTES - TENSOR_MAP_PAYLOAD_BYTES],
        );
    }

    fn make_map(
        bytes: Vec<u8>,
        global_shape: Vec<usize>,
        global_strides: Vec<usize>,
        box_shape: Vec<usize>,
        element_bits: usize,
        fp4_shared_layout: Option<Fp4SharedLayout>,
        swizzle_bytes: Option<usize>,
    ) -> (PhysicalMemory, RuntimeTensorMap) {
        let rank = global_shape.len();
        try_make_map(
            bytes,
            global_shape,
            global_strides,
            box_shape,
            vec![1; rank],
            element_bits,
            fp4_shared_layout,
            swizzle_bytes,
        )
        .unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn try_make_map(
        bytes: Vec<u8>,
        global_shape: Vec<usize>,
        global_strides: Vec<usize>,
        box_shape: Vec<usize>,
        element_strides: Vec<usize>,
        element_bits: usize,
        fp4_shared_layout: Option<Fp4SharedLayout>,
        swizzle_bytes: Option<usize>,
    ) -> Result<(PhysicalMemory, RuntimeTensorMap), EngineError> {
        let global = GlobalMemory::new();
        let allocation = global.allocate_from_bytes(bytes).unwrap();
        let view = global.full_view(allocation).unwrap();
        let element_type = match element_bits {
            4 => TensorMapElementType::Float4E2M1Fn,
            8 => TensorMapElementType::U8,
            16 => TensorMapElementType::U16,
            32 => TensorMapElementType::U32,
            64 => TensorMapElementType::U64,
            _ => TensorMapElementType::Bool,
        };
        let tensor_map = RuntimeTensorMap::new(
            view,
            global_shape,
            global_strides,
            box_shape,
            element_strides,
            element_bits,
            element_type,
            fp4_shared_layout,
            swizzle_bytes,
            TensorMapFillMode::Zero,
        )?;
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        Ok((PhysicalMemory::with_global(topology, global), tensor_map))
    }

    fn reduction_map(element_type: TensorMapElementType) -> RuntimeTensorMap {
        let global = GlobalMemory::new();
        let allocation = global.allocate_zeroed(16).unwrap();
        let view = global.full_view(allocation).unwrap();
        let element_bits = element_type.bits();
        let element_count = 128 / element_bits;
        RuntimeTensorMap::new(
            view,
            vec![element_count],
            vec![],
            vec![element_count],
            vec![1],
            element_bits,
            element_type,
            None,
            None,
            TensorMapFillMode::Zero,
        )
        .unwrap()
    }

    fn shared_destination(
        physical: &PhysicalMemory,
        byte_len: usize,
    ) -> (PhysicalPtr, RuntimeBuffer, PhysicalAllocationId) {
        let topology = physical.topology();
        let owner = CtaId::new(topology, 0, 0).unwrap();
        let allocation = physical
            .shared()
            .allocate_cta_zeroed(owner, byte_len)
            .unwrap();
        let physical_allocation = PhysicalAllocationId::from(allocation.allocation());
        let buffer = RuntimeBuffer::Shared {
            allocations: Arc::new(vec![allocation]),
            byte_offset: 0,
            byte_len,
            backing_byte_len: byte_len,
            virtual_base: 0,
        };
        (
            PhysicalPtr::new(buffer.clone(), WarpValue::splat(0_i64), 1),
            buffer,
            physical_allocation,
        )
    }

    fn async_issue_operation(context: &WarpContext) -> OperationContext {
        OperationContext::new(
            DynamicOpId::new(0, context.global_warp_id(), 0, StaticOpId::new(1), []),
            OperationKind::AsyncIssue,
            WarpMask::from_lanes([0]).unwrap(),
        )
    }

    fn footprint_bytes(access: &PhysicalAccessBatch) -> BTreeSet<(PhysicalAllocationId, usize)> {
        access
            .lanes()
            .iter()
            .flat_map(|lane| lane.footprint().spans())
            .flat_map(|span| {
                (span.byte_offset()..span.byte_end())
                    .map(move |byte_offset| (span.allocation(), byte_offset))
            })
            .collect()
    }

    fn footprint_bytes_in_space(
        accesses: &[PhysicalAccessBatch],
        space: PhysicalAccessSpace,
    ) -> BTreeSet<(PhysicalAllocationId, usize)> {
        accesses
            .iter()
            .filter(|access| access.descriptor().space() == space)
            .flat_map(footprint_bytes)
            .collect()
    }

    #[test]
    fn tensor_copy_round_trips_rank_dtype_swizzle_and_pointer_phase_matrix() {
        let cases = [
            (1_usize, 8_usize, None, 0_usize),
            (2, 16, Some(32), 1),
            (3, 32, Some(64), 3),
            (4, 64, Some(128), 7),
            (5, 8, Some(128), 5),
        ];

        for (rank, element_bits, swizzle_bytes, pointer_phase) in cases {
            let inner_elements = 128 / element_bits;
            let mut shape = vec![inner_elements];
            shape.extend(std::iter::repeat_n(2, rank - 1));
            let mut global_strides = Vec::with_capacity(rank.saturating_sub(1));
            let mut byte_len = 16_usize;
            for &extent in shape.iter().skip(1) {
                global_strides.push(byte_len);
                byte_len *= extent;
            }
            let source_bytes = (0..byte_len)
                .map(|index| (index as u8).wrapping_mul(29).wrapping_add(17))
                .collect::<Vec<_>>();
            let global = GlobalMemory::new();
            let source_allocation = global
                .allocate_from_bytes(
                    [vec![0x91; 32], source_bytes.clone(), vec![0x73; 16]].concat(),
                )
                .unwrap();
            let source_view = global.view(source_allocation, 32, byte_len).unwrap();
            let element_type = match element_bits {
                8 => TensorMapElementType::U8,
                16 => TensorMapElementType::U16,
                32 => TensorMapElementType::U32,
                64 => TensorMapElementType::U64,
                _ => unreachable!(),
            };
            let source_map = RuntimeTensorMap::new(
                source_view,
                shape.clone(),
                global_strides.clone(),
                shape.clone(),
                vec![1; rank],
                element_bits,
                element_type,
                None,
                swizzle_bytes,
                TensorMapFillMode::Zero,
            )
            .unwrap();

            let topology = LaunchTopology::new(1, 1, 1).unwrap();
            let physical = PhysicalMemory::with_global(topology, global.clone());
            let context = topology.warp_contexts().next().unwrap();
            let owner = CtaId::new(topology, 0, 0).unwrap();
            let pointer_base = pointer_phase * 128;
            let outer_count = 1_usize << rank.saturating_sub(1);
            let row_stride = swizzle_bytes.unwrap_or(16);
            let shared_byte_len = pointer_base + outer_count * row_stride;
            let shared_view = physical
                .shared()
                .allocate_cta_zeroed(owner, shared_byte_len)
                .unwrap();
            let shared_allocation = PhysicalAllocationId::from(shared_view.allocation());
            let shared_buffer = RuntimeBuffer::Shared {
                allocations: Arc::new(vec![shared_view]),
                byte_offset: 0,
                byte_len: shared_byte_len,
                backing_byte_len: shared_byte_len,
                virtual_base: 0,
            };
            let shared_pointer = PhysicalPtr::new(
                shared_buffer.clone(),
                WarpValue::splat(pointer_base as i64),
                1,
            );
            let origin = vec![0_i64; rank];
            let operation = async_issue_operation(&context);
            let (accesses, delivered) = RawTmaG2cTransferPlan::new(
                &context,
                &shared_pointer,
                &source_map,
                &origin,
                1,
                false,
            )
            .unwrap()
            .accesses_with(
                &operation,
                &context,
                &source_map,
                TmaSourceAccessPlan::Units,
            )
            .unwrap();
            assert_eq!(delivered as usize, byte_len);
            let expected_source_footprint = (32..32 + byte_len)
                .map(|offset| (PhysicalAllocationId::from(source_allocation), offset))
                .collect::<BTreeSet<_>>();
            assert_eq!(
                footprint_bytes_in_space(&accesses, PhysicalAccessSpace::Global),
                expected_source_footprint
            );

            RawTmaG2cTransferPlan::new(&context, &shared_pointer, &source_map, &origin, 1, false)
                .unwrap()
                .execute_with_report(&physical, &context, &source_map, 0)
                .map(|(delivery, _)| delivery)
                .unwrap();
            let actual_shared =
                read_runtime_bytes(&physical, &context, &shared_buffer, 0, 0, shared_byte_len)
                    .unwrap();
            let mut expected_shared = vec![0_u8; shared_byte_len];
            let mut expected_shared_footprint = BTreeSet::new();
            for outer in 0..outer_count {
                for byte_in_row in 0..16 {
                    let relative = if let Some(swizzle_bytes) = swizzle_bytes {
                        let groups = swizzle_bytes / 16;
                        let row_shift = match groups {
                            2 => 2,
                            4 => 1,
                            8 => 0,
                            _ => unreachable!(),
                        };
                        let atom = ((outer >> row_shift) + pointer_phase) % groups;
                        outer * swizzle_bytes + atom * 16 + byte_in_row
                    } else {
                        outer * 16 + byte_in_row
                    };
                    let destination_offset = pointer_base + relative;
                    expected_shared[destination_offset] = source_bytes[outer * 16 + byte_in_row];
                    expected_shared_footprint.insert((shared_allocation, destination_offset));
                }
            }
            assert_eq!(actual_shared, expected_shared);
            assert_eq!(
                footprint_bytes_in_space(&accesses, PhysicalAccessSpace::Shared),
                expected_shared_footprint
            );

            let destination_allocation = global.allocate_zeroed(32 + byte_len + 16).unwrap();
            let destination_map = RuntimeTensorMap::new(
                global.view(destination_allocation, 32, byte_len).unwrap(),
                shape.clone(),
                global_strides.clone(),
                shape,
                vec![1; rank],
                element_bits,
                element_type,
                None,
                swizzle_bytes,
                TensorMapFillMode::Zero,
            )
            .unwrap();
            let writes =
                RawTmaS2gTransferPlan::new(&context, &shared_pointer, &destination_map, &origin)
                    .unwrap()
                    .execute(&physical, &context, &destination_map, None)
                    .unwrap();
            crate::memory::publish_deferred_global_writes(&writes).unwrap();
            let round_trip = global
                .snapshot_allocation_bytes(destination_allocation)
                .unwrap();
            assert_eq!(&round_trip[32..32 + byte_len], source_bytes.as_slice());
        }
    }

    #[test]
    fn tensor_copy_oob_nan_fill_and_source_footprint_match_scalar_oracle() {
        for (element_bits, element_type) in [
            (16_usize, TensorMapElementType::F16),
            (32, TensorMapElementType::F32),
            (64, TensorMapElementType::F64),
        ] {
            let element_bytes = element_bits / 8;
            let element_count = 16 / element_bytes;
            let source_bytes = (0..16)
                .map(|index| (index as u8).wrapping_mul(11).wrapping_add(3))
                .collect::<Vec<_>>();
            let global = GlobalMemory::new();
            let allocation = global.allocate_from_bytes(source_bytes.clone()).unwrap();
            let tensor_map = RuntimeTensorMap::new(
                global.full_view(allocation).unwrap(),
                vec![element_count],
                vec![],
                vec![element_count],
                vec![1],
                element_bits,
                element_type,
                None,
                None,
                TensorMapFillMode::OobNan,
            )
            .unwrap();
            let topology = LaunchTopology::new(1, 1, 1).unwrap();
            let physical = PhysicalMemory::with_global(topology, global);
            let context = topology.warp_contexts().next().unwrap();
            let (destination, destination_buffer, _) = shared_destination(&physical, 16);
            let origin = [i64::try_from(element_count / 2).unwrap()];
            let operation = async_issue_operation(&context);
            let (accesses, _) =
                RawTmaG2cTransferPlan::new(&context, &destination, &tensor_map, &origin, 1, false)
                    .unwrap()
                    .accesses_with(
                        &operation,
                        &context,
                        &tensor_map,
                        TmaSourceAccessPlan::Units,
                    )
                    .unwrap();
            RawTmaG2cTransferPlan::new(&context, &destination, &tensor_map, &origin, 1, false)
                .unwrap()
                .execute_with_report(&physical, &context, &tensor_map, 0)
                .map(|(delivery, _)| delivery)
                .unwrap();
            let actual =
                read_runtime_bytes(&physical, &context, &destination_buffer, 0, 0, 16).unwrap();
            let valid_bytes = 8_usize;
            let mut expected = source_bytes[valid_bytes..].to_vec();
            expected
                .extend(std::iter::repeat_n(0x7ff7_u16.to_le_bytes(), valid_bytes / 2).flatten());
            assert_eq!(actual, expected);
            let expected_footprint = (valid_bytes..16)
                .map(|offset| (PhysicalAllocationId::from(allocation), offset))
                .collect::<BTreeSet<_>>();
            assert_eq!(
                footprint_bytes_in_space(&accesses, PhysicalAccessSpace::Global),
                expected_footprint
            );
        }
    }

    #[test]
    fn fp4_tensor_store_preserves_odd_origin_nibbles() {
        let global = GlobalMemory::new();
        let allocation = global.allocate_zeroed(16).unwrap();
        let tensor_map = RuntimeTensorMap::new(
            global.full_view(allocation).unwrap(),
            vec![32, 1],
            vec![16],
            vec![32, 1],
            vec![1, 1],
            4,
            TensorMapElementType::Float4E2M1Fn,
            Some(Fp4SharedLayout::Align8Packed),
            None,
            TensorMapFillMode::Zero,
        )
        .unwrap();
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let physical = PhysicalMemory::with_global(topology, global.clone());
        let context = topology.warp_contexts().next().unwrap();
        let (source, source_buffer, _) = shared_destination(&physical, 16);
        let source_bytes = (0..16)
            .map(|index| (index as u8).wrapping_mul(19).wrapping_add(0x21))
            .collect::<Vec<_>>();
        write_runtime_bytes(&physical, &context, &source_buffer, 0, 0, &source_bytes).unwrap();

        let writes = RawTmaS2gTransferPlan::new(&context, &source, &tensor_map, &[1, 0])
            .unwrap()
            .execute(&physical, &context, &tensor_map, None)
            .unwrap();
        crate::memory::publish_deferred_global_writes(&writes).unwrap();
        let actual = global.snapshot_allocation_bytes(allocation).unwrap();
        let mut expected = vec![0_u8; 16];
        for local_element in 0..31 {
            let source_nibble =
                (source_bytes[local_element / 2] >> ((local_element % 2) * 4)) & 0x0f;
            let global_element = local_element + 1;
            expected[global_element / 2] |= source_nibble << ((global_element % 2) * 4);
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn tensor_copy_multicast_delivers_identical_bytes_and_distinct_footprints() {
        let source_bytes = (0_u8..16).map(|value| value ^ 0x5a).collect::<Vec<_>>();
        let global = GlobalMemory::new();
        let allocation = global.allocate_from_bytes(source_bytes.clone()).unwrap();
        let tensor_map = RuntimeTensorMap::new(
            global.full_view(allocation).unwrap(),
            vec![16],
            vec![],
            vec![16],
            vec![1],
            8,
            TensorMapElementType::U8,
            None,
            None,
            TensorMapFillMode::Zero,
        )
        .unwrap();
        let topology = LaunchTopology::new(1, 2, 1).unwrap();
        let physical = PhysicalMemory::with_global(topology, global);
        let context = topology.warp_contexts().next().unwrap();
        let mut allocations = Vec::new();
        for cta_rank in 0..2 {
            let owner = CtaId::new(topology, 0, cta_rank).unwrap();
            allocations.push(physical.shared().allocate_cta_zeroed(owner, 16).unwrap());
        }
        let allocation_ids = allocations
            .iter()
            .map(|view| PhysicalAllocationId::from(view.allocation()))
            .collect::<Vec<_>>();
        let destination_buffer = RuntimeBuffer::Shared {
            allocations: Arc::new(allocations.clone()),
            byte_offset: 0,
            byte_len: 16,
            backing_byte_len: 16,
            virtual_base: 0,
        };
        let destination = PhysicalPtr::new(destination_buffer, WarpValue::splat(0_i64), 1);
        let operation = async_issue_operation(&context);
        let (accesses, delivered) =
            RawTmaG2cTransferPlan::new(&context, &destination, &tensor_map, &[0], 0b11, true)
                .unwrap()
                .accesses_with(
                    &operation,
                    &context,
                    &tensor_map,
                    TmaSourceAccessPlan::Units,
                )
                .unwrap();
        assert_eq!(delivered, 16);
        let result =
            RawTmaG2cTransferPlan::new(&context, &destination, &tensor_map, &[0], 0b11, true)
                .unwrap()
                .execute_with_report(&physical, &context, &tensor_map, 0)
                .map(|(delivery, _)| delivery)
                .unwrap();
        assert_eq!(result.target_ctas(), &[0, 1]);
        for (cta_rank, allocation) in allocations.iter().enumerate() {
            let owner = CtaId::new(topology, 0, cta_rank).unwrap();
            let view = physical.shared().full_cta_view(owner, allocation).unwrap();
            assert_eq!(
                physical.shared().read_bytes(&view, 0, 16).unwrap(),
                source_bytes
            );
        }
        let expected_footprint = allocation_ids
            .into_iter()
            .flat_map(|allocation| (0..16).map(move |offset| (allocation, offset)))
            .collect::<BTreeSet<_>>();
        assert_eq!(
            footprint_bytes_in_space(&accesses, PhysicalAccessSpace::Shared),
            expected_footprint
        );
    }

    #[test]
    fn tensor_reduction_applies_each_in_bounds_element_once() {
        let global = GlobalMemory::new();
        let initial = [10_u32, 20, 30, 40];
        let allocation = global
            .allocate_from_bytes(
                initial
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let tensor_map = RuntimeTensorMap::new(
            global.full_view(allocation).unwrap(),
            vec![4],
            vec![],
            vec![4],
            vec![1],
            32,
            TensorMapElementType::U32,
            None,
            None,
            TensorMapFillMode::Zero,
        )
        .unwrap();
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let physical = PhysicalMemory::with_global(topology, global.clone());
        let context = topology.warp_contexts().next().unwrap();
        let (source, source_buffer, _) = shared_destination(&physical, 16);
        let contributions = [1_u32, 2, 3, 99];
        let contribution_bytes = contributions
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        write_runtime_bytes(
            &physical,
            &context,
            &source_buffer,
            0,
            0,
            &contribution_bytes,
        )
        .unwrap();
        let operation = async_issue_operation(&context);
        let (_, destination_accesses) =
            RawTmaS2gTransferPlan::new(&context, &source, &tensor_map, &[1])
                .unwrap()
                .accesses(
                    &operation,
                    &context,
                    &tensor_map,
                    PhysicalAccessKind::AtomicReadModifyWrite,
                )
                .unwrap();
        assert!(destination_accesses.iter().all(|access| {
            access.descriptor().memory_semantics() == MemoryAccessSemantics::async_reduction()
        }));
        let writes = RawTmaS2gTransferPlan::new(&context, &source, &tensor_map, &[1])
            .unwrap()
            .execute(
                &physical,
                &context,
                &tensor_map,
                Some(RawTmaReductionOp::Add),
            )
            .unwrap();
        crate::memory::publish_deferred_global_writes(&writes).unwrap();
        let actual = global.snapshot_allocation_bytes(allocation).unwrap();
        let actual = actual
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(actual, vec![10, 21, 32, 43]);
    }

    #[test]
    fn reduction_mapping_covers_every_valid_ptx_operation_type_pair() {
        use DeferredGlobalReduction as Reduction;

        let cases = [
            (
                RawTmaReductionOp::Add,
                TensorMapElementType::U32,
                Reduction::AddU32,
            ),
            (
                RawTmaReductionOp::Add,
                TensorMapElementType::I32,
                Reduction::AddI32,
            ),
            (
                RawTmaReductionOp::Add,
                TensorMapElementType::U64,
                Reduction::AddU64,
            ),
            (
                RawTmaReductionOp::Add,
                TensorMapElementType::F32,
                Reduction::AddF32,
            ),
            (
                RawTmaReductionOp::Add,
                TensorMapElementType::F16,
                Reduction::AddF16,
            ),
            (
                RawTmaReductionOp::Add,
                TensorMapElementType::Bf16,
                Reduction::AddBf16,
            ),
            (
                RawTmaReductionOp::Min,
                TensorMapElementType::U32,
                Reduction::MinU32,
            ),
            (
                RawTmaReductionOp::Min,
                TensorMapElementType::I32,
                Reduction::MinI32,
            ),
            (
                RawTmaReductionOp::Min,
                TensorMapElementType::U64,
                Reduction::MinU64,
            ),
            (
                RawTmaReductionOp::Min,
                TensorMapElementType::I64,
                Reduction::MinI64,
            ),
            (
                RawTmaReductionOp::Min,
                TensorMapElementType::F16,
                Reduction::MinF16,
            ),
            (
                RawTmaReductionOp::Min,
                TensorMapElementType::Bf16,
                Reduction::MinBf16,
            ),
            (
                RawTmaReductionOp::Max,
                TensorMapElementType::U32,
                Reduction::MaxU32,
            ),
            (
                RawTmaReductionOp::Max,
                TensorMapElementType::I32,
                Reduction::MaxI32,
            ),
            (
                RawTmaReductionOp::Max,
                TensorMapElementType::U64,
                Reduction::MaxU64,
            ),
            (
                RawTmaReductionOp::Max,
                TensorMapElementType::I64,
                Reduction::MaxI64,
            ),
            (
                RawTmaReductionOp::Max,
                TensorMapElementType::F16,
                Reduction::MaxF16,
            ),
            (
                RawTmaReductionOp::Max,
                TensorMapElementType::Bf16,
                Reduction::MaxBf16,
            ),
            (
                RawTmaReductionOp::Inc,
                TensorMapElementType::U32,
                Reduction::IncU32,
            ),
            (
                RawTmaReductionOp::Dec,
                TensorMapElementType::U32,
                Reduction::DecU32,
            ),
            (
                RawTmaReductionOp::And,
                TensorMapElementType::U32,
                Reduction::AndB32,
            ),
            (
                RawTmaReductionOp::And,
                TensorMapElementType::U64,
                Reduction::AndB64,
            ),
            (
                RawTmaReductionOp::Or,
                TensorMapElementType::U32,
                Reduction::OrB32,
            ),
            (
                RawTmaReductionOp::Or,
                TensorMapElementType::U64,
                Reduction::OrB64,
            ),
            (
                RawTmaReductionOp::Xor,
                TensorMapElementType::U32,
                Reduction::XorB32,
            ),
            (
                RawTmaReductionOp::Xor,
                TensorMapElementType::U64,
                Reduction::XorB64,
            ),
        ];

        for (operation, element_type, expected) in cases {
            assert_eq!(
                reduction_map(element_type).reduction(operation).unwrap(),
                expected,
                "operation={operation:?}, dtype={element_type}"
            );
        }
    }

    #[test]
    fn reduction_mapping_rejects_invalid_ptx_operation_type_pairs() {
        for (operation, element_type) in [
            (RawTmaReductionOp::Add, TensorMapElementType::I64),
            (RawTmaReductionOp::Min, TensorMapElementType::F32),
            (RawTmaReductionOp::Max, TensorMapElementType::F32),
            (RawTmaReductionOp::Inc, TensorMapElementType::I32),
            (RawTmaReductionOp::And, TensorMapElementType::U16),
        ] {
            let error = reduction_map(element_type)
                .reduction(operation)
                .unwrap_err();
            assert!(
                error.to_string().contains("is invalid for TensorMap dtype"),
                "operation={operation:?}, dtype={element_type}: {error}"
            );
        }
    }

    #[test]
    fn fp4_geometry_and_transaction_accounting_match_shared_layout() {
        let global = GlobalMemory::new();
        let allocation = global.allocate_zeroed(128 * 6).unwrap();
        let error = RuntimeTensorMap::new_with_layout(
            global.full_view(allocation).unwrap(),
            vec![8, 3, 2],
            vec![128, 384],
            vec![4, 1, 1],
            vec![1; 3],
            4,
            TensorMapElementType::Float4E2M1Fn,
            Some(Fp4SharedLayout::Align16Padded),
            None,
            SwizzleAtomicity::B16,
            TensorMapFillMode::Zero,
            Some(16),
            None,
        )
        .err()
        .expect("padded interleave remains unmodeled");
        assert!(matches!(
            error.kind(),
            crate::EngineErrorKind::AnalysisIncomplete {
                kind: "tma_padded_fp4_interleave_unmodeled"
            }
        ));
        for bytes in [16, 32] {
            let geometry = tensor_map_geometry_from_metadata(
                &[3, 2, 1],
                bytes * 8,
                Some(Fp4SharedLayout::Align8Packed),
            )
            .unwrap();
            assert_eq!(geometry.packed_elements, 1);
            assert_eq!(geometry.unit_bytes, bytes);
            assert_eq!(geometry.unit_stride_bytes, bytes);
            assert_eq!(geometry.inner_units, 3);
            assert_eq!(geometry.inner_row_bytes, 3 * bytes);
            assert_eq!(geometry.outer_count, 2);
        }
        let (_, align8) = make_map(
            vec![0; 128],
            vec![128, 2],
            vec![64],
            vec![128, 2],
            4,
            Some(Fp4SharedLayout::Align8Packed),
            Some(64),
        );
        assert_eq!(
            tensor_map_geometry(&align8).unwrap(),
            TensorMapGeometry {
                packed_elements: 2,
                unit_bytes: 1,
                unit_stride_bytes: 1,
                inner_units: 64,
                inner_row_bytes: 64,
                outer_count: 2,
            }
        );
        assert_eq!(tensor_map_transaction_bytes(1).unwrap(), 1);

        let (_, align16) = make_map(
            vec![0; 128],
            vec![128, 2],
            vec![64],
            vec![128, 2],
            4,
            Some(Fp4SharedLayout::Align16Padded),
            Some(128),
        );
        assert_eq!(
            tensor_map_geometry(&align16).unwrap(),
            TensorMapGeometry {
                packed_elements: 16,
                unit_bytes: 8,
                unit_stride_bytes: 16,
                inner_units: 8,
                inner_row_bytes: 128,
                outer_count: 2,
            }
        );
        assert_eq!(tensor_map_transaction_bytes(8).unwrap(), 8);
    }

    #[test]
    fn rank_coordinates_and_swizzle_are_physical_layout_operations() {
        let (_, rank3) = make_map(
            vec![0; 96],
            vec![4, 3, 2],
            vec![16, 48],
            vec![4, 3, 2],
            32,
            None,
            None,
        );
        let outer = tensor_map_outer_coordinates(&rank3, 5);
        assert_eq!(outer, vec![0, 2, 1]);
        let coordinates = tensor_map_global_coordinates(&rank3, &[0, 0, 0], 2, &outer).unwrap();
        assert_eq!(coordinates, vec![2, 2, 1]);
        assert_eq!(
            tensor_map_global_byte_offset(&rank3, &coordinates).unwrap(),
            (88, 0)
        );

        let (_, swizzled) = make_map(
            vec![0; 256],
            vec![8, 8],
            vec![32],
            vec![8, 8],
            32,
            None,
            Some(32),
        );
        assert_eq!(
            tensor_map_shared_byte_offset(&swizzled, 4, 0, 32, 0).unwrap(),
            144
        );

        let (_, align16) = make_map(
            vec![0; 128],
            vec![128, 2],
            vec![64],
            vec![128, 2],
            4,
            Some(Fp4SharedLayout::Align16Padded),
            Some(128),
        );
        assert_eq!(
            tensor_map_shared_byte_offset(&align16, 1, 0, 128, 0).unwrap(),
            144
        );
        assert_eq!(
            tensor_map_shared_byte_offset(&align16, 1, 16, 128, 0).unwrap(),
            128
        );
    }

    #[test]
    fn repeated_read_payloads_zero_fill_out_of_bounds_units() {
        let (physical, tensor_map) = make_map(
            (10_u8..42).collect(),
            vec![16, 2],
            vec![16],
            vec![16, 4],
            8,
            None,
            None,
        );
        let template = &tensor_map.transfer_template;
        for (origin, expected) in [([0, 0], 12), ([0, 2], 0)] {
            let source = template.bind_global(&tensor_map, &origin).unwrap();
            let (payload, _) = materialize_raw_tma_g2c_payload(
                &physical,
                &tensor_map,
                &source.runs,
                template.payload_len,
                template.geometry.unit_bytes,
                tensor_map.fill_mode,
                0,
            )
            .unwrap();
            assert_eq!(payload.len(), 64);
            assert_eq!(payload[2], expected);
            assert_eq!(&payload[32..], &[0; 32]);
        }
    }

    #[test]
    fn fp4_read_codec_matches_packed_and_padded_shared_layouts() {
        let read = |physical: &PhysicalMemory, map: &RuntimeTensorMap| {
            let template = &map.transfer_template;
            let source = template.bind_global(map, &[0, 0]).unwrap();
            materialize_raw_tma_g2c_payload(
                physical,
                map,
                &source.runs,
                template.payload_len,
                template.geometry.unit_bytes,
                map.fill_mode,
                0,
            )
            .unwrap()
            .0
        };
        let (physical, align8) = make_map(
            [vec![0x21, 0x43], vec![0; 14]].concat(),
            vec![32, 1],
            vec![16],
            vec![32, 1],
            4,
            Some(Fp4SharedLayout::Align8Packed),
            None,
        );
        assert_eq!(read(&physical, &align8)[..1], [0x21]);

        let packed = vec![0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe];
        let (physical, align16) = make_map(
            [packed.clone(), vec![0; 56]].concat(),
            vec![128, 1],
            vec![64],
            vec![128, 1],
            4,
            Some(Fp4SharedLayout::Align16Padded),
            None,
        );
        assert_eq!(&read(&physical, &align16)[..8], packed.as_slice());
    }

    #[test]
    fn element_stride_controls_traversal_count_and_coordinates() {
        let (_, tensor_map) = try_make_map(
            vec![0; 16 * 5],
            vec![16, 5],
            vec![16],
            vec![16, 5],
            vec![7, 2],
            8,
            None,
            None,
        )
        .unwrap();

        assert_eq!(tensor_map.element_strides, vec![1, 2]);
        assert_eq!(tensor_map.traversal_shape, vec![16, 3]);
        assert_eq!(
            tensor_map_geometry(&tensor_map).unwrap(),
            TensorMapGeometry {
                packed_elements: 1,
                unit_bytes: 1,
                unit_stride_bytes: 1,
                inner_units: 16,
                inner_row_bytes: 16,
                outer_count: 3,
            }
        );
        let outer = tensor_map_outer_coordinates(&tensor_map, 2);
        assert_eq!(outer, vec![0, 2]);
        assert_eq!(
            tensor_map_global_coordinates(&tensor_map, &[0, 0], 7, &outer).unwrap(),
            vec![7, 4]
        );
    }

    #[test]
    fn tensor_copy_observes_outer_element_stride_in_both_directions() {
        let source_bytes = (0_u8..80).collect::<Vec<_>>();
        let (physical, source_map) = try_make_map(
            source_bytes.clone(),
            vec![16, 5],
            vec![16],
            vec![16, 5],
            vec![1, 2],
            8,
            None,
            None,
        )
        .unwrap();
        let context = physical.topology().warp_contexts().next().unwrap();
        let (shared_pointer, shared_buffer, _) = shared_destination(&physical, 48);
        let operation = async_issue_operation(&context);
        let (accesses, delivered) =
            RawTmaG2cTransferPlan::new(&context, &shared_pointer, &source_map, &[0, 0], 1, false)
                .unwrap()
                .accesses_with(
                    &operation,
                    &context,
                    &source_map,
                    TmaSourceAccessPlan::Units,
                )
                .unwrap();
        assert_eq!(delivered, 48);
        RawTmaG2cTransferPlan::new(&context, &shared_pointer, &source_map, &[0, 0], 1, false)
            .unwrap()
            .execute_with_report(&physical, &context, &source_map, 0)
            .map(|(delivery, _)| delivery)
            .unwrap();

        let expected_payload = [
            &source_bytes[0..16],
            &source_bytes[32..48],
            &source_bytes[64..80],
        ]
        .concat();
        assert_eq!(
            read_runtime_bytes(&physical, &context, &shared_buffer, 0, 0, 48).unwrap(),
            expected_payload
        );
        let source_allocation = PhysicalAllocationId::from(source_map.view.allocation());
        let expected_footprint = [0_usize, 32, 64]
            .into_iter()
            .flat_map(|row| (row..row + 16).map(move |offset| (source_allocation, offset)))
            .collect::<BTreeSet<_>>();
        assert_eq!(
            footprint_bytes_in_space(&accesses, PhysicalAccessSpace::Global),
            expected_footprint
        );

        let destination_allocation = physical.global().allocate_zeroed(80).unwrap();
        let destination_map = RuntimeTensorMap::new(
            physical.global().full_view(destination_allocation).unwrap(),
            vec![16, 5],
            vec![16],
            vec![16, 5],
            vec![1, 2],
            8,
            TensorMapElementType::U8,
            None,
            None,
            TensorMapFillMode::Zero,
        )
        .unwrap();
        let writes =
            RawTmaS2gTransferPlan::new(&context, &shared_pointer, &destination_map, &[0, 0])
                .unwrap()
                .execute(&physical, &context, &destination_map, None)
                .unwrap();
        crate::memory::publish_deferred_global_writes(&writes).unwrap();
        let mut expected_destination = vec![0_u8; 80];
        expected_destination[0..16].copy_from_slice(&expected_payload[0..16]);
        expected_destination[32..48].copy_from_slice(&expected_payload[16..32]);
        expected_destination[64..80].copy_from_slice(&expected_payload[32..48]);
        assert_eq!(
            physical
                .global()
                .snapshot_allocation_bytes(destination_allocation)
                .unwrap(),
            expected_destination
        );
    }

    #[test]
    fn descriptor_validation_rejects_hardware_illegal_ranges() {
        let invalid_box = try_make_map(
            vec![0; 512],
            vec![16, 2],
            vec![16],
            vec![16, 257],
            vec![1, 1],
            8,
            None,
            None,
        )
        .err()
        .unwrap();
        assert!(invalid_box.to_string().contains("1..=256"));

        let invalid_element_stride = try_make_map(
            vec![0; 32],
            vec![16, 2],
            vec![16],
            vec![16, 2],
            vec![1, 9],
            8,
            None,
            None,
        )
        .err()
        .unwrap();
        assert!(invalid_element_stride.to_string().contains("1..=8"));

        let invalid_global_stride = try_make_map(
            vec![0; 32],
            vec![16, 2],
            vec![24],
            vec![16, 2],
            vec![1, 1],
            8,
            None,
            None,
        )
        .err()
        .unwrap();
        assert!(invalid_global_stride
            .to_string()
            .contains("multiples of 16 below 2^40"));

        let oversized_global_stride = try_make_map(
            vec![0; 32],
            vec![16, 2],
            vec![1_usize << 40],
            vec![16, 2],
            vec![1, 1],
            8,
            None,
            None,
        )
        .err()
        .unwrap();
        assert!(oversized_global_stride
            .to_string()
            .contains("multiples of 16 below 2^40"));

        let overlapping_global_stride = try_make_map(
            vec![0; 64],
            vec![16, 2, 2],
            vec![16, 16],
            vec![16, 2, 2],
            vec![1, 1, 1],
            8,
            None,
            None,
        )
        .err()
        .unwrap();
        assert!(overlapping_global_stride
            .to_string()
            .contains("overlaps the prior 32-byte span"));

        let invalid_inner_transfer = try_make_map(
            vec![0; 16],
            vec![16],
            vec![],
            vec![15],
            vec![1],
            8,
            None,
            None,
        )
        .err()
        .unwrap();
        assert!(invalid_inner_transfer
            .to_string()
            .contains("multiple of 16 bytes"));

        let invalid_packed_shape = try_make_map(
            vec![0; 64],
            vec![64, 2],
            vec![32],
            vec![128, 1],
            vec![1, 1],
            4,
            Some(Fp4SharedLayout::Align16Padded),
            Some(128),
        )
        .err()
        .unwrap();
        assert!(invalid_packed_shape
            .to_string()
            .contains("global dimension zero to be a multiple of 128"));

        if usize::BITS > 32 {
            let invalid_global_dimension = try_make_map(
                vec![0; 16],
                vec![(1_u64 << 32) as usize + 1],
                vec![],
                vec![16],
                vec![1],
                8,
                None,
                None,
            )
            .err()
            .unwrap();
            assert!(invalid_global_dimension
                .to_string()
                .contains("at most 2^32"));
        }
    }

    #[test]
    fn descriptor_accepts_permuted_nonoverlapping_outer_axes() {
        let (_, tensor_map) = try_make_map(
            vec![0; 64 * 2 * 128 * 2],
            vec![128, 64, 2],
            vec![512, 256],
            vec![64, 64, 1],
            vec![1, 1, 1],
            16,
            None,
            Some(128),
        )
        .unwrap();

        assert_eq!(tensor_map.global_strides, vec![512, 256]);
    }
}
