//! Engine implementation of the v2 memory-instruction specializations.

use crate::runtime::abi_transport::PtxAddressSpace;
use std::marker::PhantomData;

use super::instruction::{
    async_instruction, instruction_variant, sync_instruction, sync_instruction_generic_args,
};
use super::mode_axis::for_each_engine_mode;
use super::transport::engine;
use super::{
    Address, DirectAddress, ExecCtx, Generic, Global, Local, MemorySpace, Shared, SharedCluster,
    SharedCta, SiteId, WarpHandle, R,
};
use crate::runtime::{
    raw_atomic_add_bf16_physical_ptr_warp, raw_atomic_add_bf16x2_physical_ptr_warp,
    raw_atomic_add_f32x2_physical_ptr_warp, raw_atomic_add_f32x4_physical_ptr_warp,
    raw_atomic_add_fp16_physical_ptr_warp, raw_atomic_add_fp16x2_physical_ptr_warp,
    raw_atomic_cas_physical_ptr_warp, raw_atomic_half_vector_physical_ptr_warp,
    raw_atomic_scalar_physical_ptr_warp, raw_ldmatrix_b16_fragments, raw_ldmatrix_b8_fragments,
    raw_load_physical_ptr_warp, raw_load_physical_ptr_warp_atomic, raw_st_bulk_zero, raw_stmatrix,
    raw_store_physical_ptr_warp, raw_store_vector_physical_ptr_warp, PointerSpace, PtxStateSpace,
    RawAtomicOperation, StmatrixDescriptor,
};
use crate::{
    EngineError, MemoryAccessClass, MemoryAccessSemantics, MemoryOrder, MemoryProxy, MemoryScope,
    OperationKind, RuntimeScalar,
};

sync_instruction_generic_args!(ld_spec, LdVariant, ld);
sync_instruction_generic_args!(st_spec, StVariant, st);
async_instruction!(atom_spec, AtomVariant, atom);
async_instruction!(red_spec, RedVariant, red);
async_instruction!(multimem_spec, MultimemVariant, multimem);
sync_instruction!(ldmatrix_spec, LdmatrixVariant, ldmatrix);
sync_instruction!(stmatrix_spec, StmatrixVariant, stmatrix);
sync_instruction!(st_bulk_spec, StBulkVariant, st_bulk);

#[path = "multimem.rs"]
mod multimem_impl;

/// Static specializations for lane-wise `ld` and `st`.
pub mod variant {
    use super::PhantomData;

    /// `multimem.ld_reduce`/`st`/`red`. The const codes are decoded by
    /// `multimem_impl::MultimemForm::decode`; operands and results travel as
    /// up to four little-endian 32-bit words.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Multimem<
        const KIND: u8,
        const TYPE: u8,
        const OP: u8,
        const VEC: u8,
        const SEM: u8,
        const SCOPE: u8,
    >;

    /// One scalar lane value not already represented by a register marker.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Bool;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct F16;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Bf16;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct F16x2;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Bf16x2;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct F32x2;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct F32x4;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct U64x2;

    /// PTX `.b8`/`.u8`: one byte in memory, zero-extended into a 32-bit
    /// register. The access width and the register width differ, so this is a
    /// carrier of its own rather than `reg::variant::U8` (which is the
    /// identity 8-bit carrier used by typed byte storage).
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct U8AsU32;
    /// PTX `.s8`: one byte in memory, sign-extended into a 32-bit register.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct S8AsI32;

    /// PTX `.b16`/`.u16`: two bytes in memory, zero-extended into a 32-bit
    /// register. Same reason as `U8AsU32`: the access width and the register
    /// width differ, so the identity carrier `reg::variant::U16` cannot serve.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct U16AsU32;
    /// PTX `.s16`: two bytes in memory, sign-extended into a 32-bit register.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct S16AsI32;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Plain;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Readonly;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Volatile;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Mmio;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct MmioRelaxed;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct MmioAcquire;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct MmioRelease;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Relaxed<Scope>(PhantomData<fn() -> Scope>);
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Acquire<Scope>(PhantomData<fn() -> Scope>);
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Release<Scope>(PhantomData<fn() -> Scope>);
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct AcqRel<Scope>(PhantomData<fn() -> Scope>);

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Cta;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Cluster;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Gpu;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Sys;

    /// Exact scalar, state-space, and memory-order form of one `ld`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Ld<T, Space, Semantics = Plain>(PhantomData<fn(T, Space, Semantics)>);

    /// Exact scalar, state-space, and memory-order form of one `st`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct St<T, Space, Semantics = Plain>(PhantomData<fn(T, Space, Semantics)>);

    /// Exact vector form of the same `st` mnemonic. `N` is part of the PTX
    /// variant, not a frontend repetition count; only v2/v4/v8 are sealed.
    /// `SINKS` marks unwritten components of a 256-bit global store.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct StVec<T, Space, const N: usize, Semantics = Plain, const SINKS: u8 = 0>(
        PhantomData<fn(T, Space, Semantics)>,
    );

    /// Only FP32 atomic forms seal the subnormal-preserving specialization.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Add<const NOFTZ: bool = false>;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct BitAnd;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct BitOr;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct BitXor;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Exchange;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Increment;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Decrement;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Minimum;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Maximum;
    /// Vector half atomics, with N independently atomic 16-bit elements.
    /// Register packing is a frontend transport detail, not an atomic unit.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct HalfVector<T, const N: usize>(PhantomData<fn() -> T>);
    /// One scalar, single-operand `atom` form.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Atom<T, Space, Operation, Semantics = Plain>(
        PhantomData<fn(T, Space, Operation, Semantics)>,
    );

    /// CAS form of the same `atom` mnemonic. A distinct marker shape closes
    /// its extra runtime operand without a nullable compare value.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct AtomCas<T, Space, Semantics = Plain>(PhantomData<fn(T, Space, Semantics)>);

    /// One scalar `red` form. It deliberately has a separate marker because
    /// forms with acquire semantics are not legal without an old-value result.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Red<T, Space, Operation, Semantics = Plain>(
        PhantomData<fn(T, Space, Operation, Semantics)>,
    );

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Ldmatrix<
        const COUNT: usize,
        const TRANSPOSE: bool,
        const SOURCE_BITS: usize = 16,
        const SIGNED: bool = false,
    >;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct StmatrixM8N8B16<Space, const COUNT: usize, const TRANSPOSE: bool>(
        PhantomData<fn() -> Space>,
    );
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct StmatrixM16N8B8<Space, const COUNT: usize>(PhantomData<fn() -> Space>);

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct StBulkZeroGeneric;
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct StBulkZeroShared;
}

mod memory_type_sealed {
    pub trait Sealed {}
}

/// Compile-time scalar carrier selected by a memory instruction variant.
#[allow(private_bounds)]
pub trait MemoryType: memory_type_sealed::Sealed {
    type Scalar: RuntimeScalar + Send + Sync + 'static;
    type Storage: RuntimeScalar + Send + Sync + 'static;

    fn decode(value: Self::Storage) -> Self::Scalar;
    fn encode(value: Self::Scalar) -> Self::Storage;

    fn decode_warp(values: R<Self::Storage>) -> R<Self::Scalar> {
        values.map(|_lane, value| Self::decode(value))
    }

    fn encode_warp(values: R<Self::Scalar>) -> R<Self::Storage> {
        values.map(|_lane, value| Self::encode(value))
    }
}

macro_rules! identity_memory_types {
    ($($marker:ty => $scalar:ty),+ $(,)?) => {
        $(
            impl memory_type_sealed::Sealed for $marker {}
            impl MemoryType for $marker {
                type Scalar = $scalar;
                type Storage = $scalar;

                fn decode(value: Self::Storage) -> Self::Scalar {
                    value
                }

                fn encode(value: Self::Scalar) -> Self::Storage {
                    value
                }

                fn decode_warp(values: R<Self::Storage>) -> R<Self::Scalar> {
                    values
                }

                fn encode_warp(values: R<Self::Scalar>) -> R<Self::Storage> {
                    values
                }
            }
        )+
    };
}

identity_memory_types!(
    super::reg::variant::I8 => i8,
    super::reg::variant::I16 => i16,
    super::reg::variant::I32 => i32,
    super::reg::variant::I64 => i64,
    super::reg::variant::U8 => u8,
    super::reg::variant::U16 => u16,
    super::reg::variant::U32 => u32,
    super::reg::variant::U64 => u64,
    super::reg::variant::B32 => u32,
    super::reg::variant::B64 => u64,
    super::reg::variant::F16 => u16,
    super::reg::variant::Bf16 => u16,
    super::reg::variant::F32 => f32,
    super::reg::variant::F64 => f64,
    variant::Bool => bool,
    variant::F16x2 => u32,
    variant::Bf16x2 => u32,
    variant::F32x2 => u64,
    variant::F32x4 => crate::F32x4,
    variant::U64x2 => crate::scalar::U64x2,
    variant::HalfVector<variant::F16, 2> => u32,
    variant::HalfVector<variant::F16, 4> => u64,
    variant::HalfVector<variant::F16, 8> => crate::scalar::U64x2,
    variant::HalfVector<variant::Bf16, 2> => u32,
    variant::HalfVector<variant::Bf16, 4> => u64,
    variant::HalfVector<variant::Bf16, 8> => crate::scalar::U64x2,
);

// PTX: "The value loaded is sign-extended to the destination register width
// for signed integers, and is zero-extended to the destination register width
// for unsigned and bit-size types" (PTX ISA 9.7.9.8). A store truncates the
// same way, so `encode` is the inverse narrowing.
impl memory_type_sealed::Sealed for variant::U8AsU32 {}
impl MemoryType for variant::U8AsU32 {
    type Scalar = u32;
    type Storage = u8;

    fn decode(value: Self::Storage) -> Self::Scalar {
        u32::from(value)
    }

    fn encode(value: Self::Scalar) -> Self::Storage {
        value as u8
    }
}

impl memory_type_sealed::Sealed for variant::S8AsI32 {}
impl MemoryType for variant::S8AsI32 {
    type Scalar = i32;
    type Storage = i8;

    fn decode(value: Self::Storage) -> Self::Scalar {
        i32::from(value)
    }

    fn encode(value: Self::Scalar) -> Self::Storage {
        value as i8
    }
}

impl memory_type_sealed::Sealed for variant::U16AsU32 {}
impl MemoryType for variant::U16AsU32 {
    type Scalar = u32;
    type Storage = u16;

    fn decode(value: Self::Storage) -> Self::Scalar {
        u32::from(value)
    }

    fn encode(value: Self::Scalar) -> Self::Storage {
        value as u16
    }
}

impl memory_type_sealed::Sealed for variant::S16AsI32 {}
impl MemoryType for variant::S16AsI32 {
    type Scalar = i32;
    type Storage = i16;

    fn decode(value: Self::Storage) -> Self::Scalar {
        i32::from(value)
    }

    fn encode(value: Self::Scalar) -> Self::Storage {
        value as i16
    }
}

impl memory_type_sealed::Sealed for variant::F16 {}
impl MemoryType for variant::F16 {
    type Scalar = f32;
    type Storage = u16;

    fn decode(value: Self::Storage) -> Self::Scalar {
        crate::fp16_bits_to_f32(value)
    }

    fn encode(value: Self::Scalar) -> Self::Storage {
        crate::f32_to_fp16_bits(value)
    }
}

impl memory_type_sealed::Sealed for variant::Bf16 {}
impl MemoryType for variant::Bf16 {
    type Scalar = f32;
    type Storage = u16;

    fn decode(value: Self::Storage) -> Self::Scalar {
        crate::bf16_bits_to_f32(value)
    }

    fn encode(value: Self::Scalar) -> Self::Storage {
        crate::f32_to_bf16_bits(value)
    }
}

pub(crate) trait SpaceVariant: PtxAddressSpace {
    fn atomic_space(
        pointer: &crate::runtime::PhysicalPtr,
        mask: crate::WarpMask,
    ) -> Result<PtxStateSpace, crate::EngineError>;
}

impl SpaceVariant for Generic {
    fn atomic_space(
        pointer: &crate::runtime::PhysicalPtr,
        mask: crate::WarpMask,
    ) -> Result<PtxStateSpace, crate::EngineError> {
        match pointer.pointer_space_for_mask(mask)? {
            PointerSpace::Global => Ok(PtxStateSpace::Global),
            PointerSpace::Shared => Ok(PtxStateSpace::Shared),
            _ => Err(crate::EngineError::message(
                "generic atom/red address must resolve to global or shared memory",
            )),
        }
    }
}
impl SpaceVariant for Global {
    fn atomic_space(
        _pointer: &crate::runtime::PhysicalPtr,
        _mask: crate::WarpMask,
    ) -> Result<PtxStateSpace, crate::EngineError> {
        Ok(Self::PTX_SPACE)
    }
}
impl SpaceVariant for Shared {
    fn atomic_space(
        _pointer: &crate::runtime::PhysicalPtr,
        _mask: crate::WarpMask,
    ) -> Result<PtxStateSpace, crate::EngineError> {
        Ok(Self::PTX_SPACE)
    }
}
impl SpaceVariant for SharedCta {
    fn atomic_space(
        _pointer: &crate::runtime::PhysicalPtr,
        _mask: crate::WarpMask,
    ) -> Result<PtxStateSpace, crate::EngineError> {
        Ok(Self::PTX_SPACE)
    }
}
impl SpaceVariant for SharedCluster {
    fn atomic_space(
        _pointer: &crate::runtime::PhysicalPtr,
        _mask: crate::WarpMask,
    ) -> Result<PtxStateSpace, crate::EngineError> {
        Ok(Self::PTX_SPACE)
    }
}
impl SpaceVariant for Local {
    fn atomic_space(
        _pointer: &crate::runtime::PhysicalPtr,
        _mask: crate::WarpMask,
    ) -> Result<PtxStateSpace, crate::EngineError> {
        Ok(Self::PTX_SPACE)
    }
}

trait AtomicSpace: SpaceVariant {}
impl AtomicSpace for Generic {}
impl AtomicSpace for Global {}
impl AtomicSpace for Shared {}
impl AtomicSpace for SharedCta {}
impl AtomicSpace for SharedCluster {}

mod static_sealed {
    pub trait Scope {}
    pub trait LoadSemantics {}
    pub trait StoreSemantics {}
}

pub(crate) trait StaticScope: static_sealed::Scope {
    const VALUE: MemoryScope;
}

macro_rules! scopes {
    ($($marker:ty => $value:ident),+ $(,)?) => {
        $(
            impl static_sealed::Scope for $marker {}
            impl StaticScope for $marker {
                const VALUE: MemoryScope = MemoryScope::$value;
            }
        )+
    };
}

scopes!(
    variant::Cta => Cta,
    variant::Cluster => Cluster,
    variant::Gpu => Gpu,
    variant::Sys => Sys,
);

trait LoadSemantics: static_sealed::LoadSemantics {
    const VALUE: MemoryAccessSemantics;
    const ATOMIC_COHERENT: bool;
    const READONLY_PROXY: bool = false;
}

trait StoreSemantics: static_sealed::StoreSemantics {
    const VALUE: MemoryAccessSemantics;
    const DIRECT_NAMED: bool;
}

macro_rules! simple_load_semantics {
    ($marker:ty, $value:expr, $atomic:expr) => {
        impl static_sealed::LoadSemantics for $marker {}
        impl LoadSemantics for $marker {
            const VALUE: MemoryAccessSemantics = $value;
            const ATOMIC_COHERENT: bool = $atomic;
        }
    };
}

macro_rules! simple_store_semantics {
    ($marker:ty, $value:expr, $direct_named:expr) => {
        impl static_sealed::StoreSemantics for $marker {}
        impl StoreSemantics for $marker {
            const VALUE: MemoryAccessSemantics = $value;
            const DIRECT_NAMED: bool = $direct_named;
        }
    };
}

simple_load_semantics!(variant::Plain, MemoryAccessSemantics::plain(), false);
impl static_sealed::LoadSemantics for variant::Readonly {}
impl LoadSemantics for variant::Readonly {
    const VALUE: MemoryAccessSemantics = MemoryAccessSemantics::plain();
    const ATOMIC_COHERENT: bool = false;
    const READONLY_PROXY: bool = true;
}
simple_load_semantics!(variant::Volatile, MemoryAccessSemantics::volatile(), true);
simple_load_semantics!(variant::Mmio, MemoryAccessSemantics::mmio(), true);
simple_store_semantics!(variant::Plain, MemoryAccessSemantics::plain(), true);
simple_store_semantics!(variant::Volatile, MemoryAccessSemantics::volatile(), false);
simple_store_semantics!(variant::Mmio, MemoryAccessSemantics::mmio(), false);
simple_load_semantics!(
    variant::MmioRelaxed,
    MemoryAccessSemantics::scoped(
        MemoryOrder::Relaxed,
        MemoryScope::Sys,
        MemoryProxy::Mmio,
        MemoryAccessClass::Atomic,
    ),
    true
);
simple_load_semantics!(
    variant::MmioAcquire,
    MemoryAccessSemantics::scoped(
        MemoryOrder::Acquire,
        MemoryScope::Sys,
        MemoryProxy::Mmio,
        MemoryAccessClass::Atomic,
    ),
    true
);
simple_store_semantics!(
    variant::MmioRelaxed,
    MemoryAccessSemantics::scoped(
        MemoryOrder::Relaxed,
        MemoryScope::Sys,
        MemoryProxy::Mmio,
        MemoryAccessClass::Atomic,
    ),
    false
);
simple_store_semantics!(
    variant::MmioRelease,
    MemoryAccessSemantics::scoped(
        MemoryOrder::Release,
        MemoryScope::Sys,
        MemoryProxy::Mmio,
        MemoryAccessClass::Atomic,
    ),
    false
);

macro_rules! scoped_load_semantics {
    ($wrapper:ident, $order:ident) => {
        impl<S: StaticScope> static_sealed::LoadSemantics for variant::$wrapper<S> {}
        impl<S: StaticScope> LoadSemantics for variant::$wrapper<S> {
            const VALUE: MemoryAccessSemantics = MemoryAccessSemantics::scoped(
                MemoryOrder::$order,
                S::VALUE,
                MemoryProxy::Generic,
                MemoryAccessClass::Atomic,
            );
            const ATOMIC_COHERENT: bool = true;
        }
    };
}

macro_rules! scoped_store_semantics {
    ($wrapper:ident, $order:ident) => {
        impl<S: StaticScope> static_sealed::StoreSemantics for variant::$wrapper<S> {}
        impl<S: StaticScope> StoreSemantics for variant::$wrapper<S> {
            const VALUE: MemoryAccessSemantics = MemoryAccessSemantics::scoped(
                MemoryOrder::$order,
                S::VALUE,
                MemoryProxy::Generic,
                MemoryAccessClass::Atomic,
            );
            const DIRECT_NAMED: bool = false;
        }
    };
}

scoped_load_semantics!(Relaxed, Relaxed);
scoped_load_semantics!(Acquire, Acquire);
scoped_store_semantics!(Relaxed, Relaxed);
scoped_store_semantics!(Release, Release);

trait AtomicSemantics {
    const VALUE: MemoryAccessSemantics;
}

impl AtomicSemantics for variant::Plain {
    const VALUE: MemoryAccessSemantics = MemoryAccessSemantics::plain();
}

macro_rules! atomic_semantics {
    ($wrapper:ident, $order:ident) => {
        impl<S: StaticScope> AtomicSemantics for variant::$wrapper<S> {
            const VALUE: MemoryAccessSemantics = MemoryAccessSemantics::scoped(
                MemoryOrder::$order,
                S::VALUE,
                MemoryProxy::Generic,
                MemoryAccessClass::Atomic,
            );
        }
    };
}

atomic_semantics!(Relaxed, Relaxed);
atomic_semantics!(Acquire, Acquire);
atomic_semantics!(Release, Release);
atomic_semantics!(AcqRel, AcqRel);

trait ReductionSemantics: AtomicSemantics {}
impl ReductionSemantics for variant::Plain {}
impl<S: StaticScope> ReductionSemantics for variant::Relaxed<S> {}
impl<S: StaticScope> ReductionSemantics for variant::Release<S> {}

/// Physical effect descriptors have one space. Generic instructions may
/// choose different spaces per lane; explicit-space instructions must not.
fn first_generic_space_mask(
    pointer: &crate::runtime::PhysicalPtr,
    declared_space: PtxStateSpace,
    mask: crate::WarpMask,
) -> Result<crate::WarpMask, EngineError> {
    if declared_space != PtxStateSpace::Generic || mask.is_empty() {
        return Ok(mask);
    }
    let space = pointer.pointer_space_at_lane(mask.first_active().unwrap())?;
    let mut bits = 0;
    for lane in mask {
        if pointer.pointer_space_at_lane(lane)? == space {
            bits |= 1 << lane;
        }
    }
    Ok(crate::WarpMask::from_bits(bits))
}

/// Wait on a declared synchronization word until its predicate holds.
///
/// One suspending operation, not a loop of loads: the polling a kernel writes
/// is the wait's mechanism, and every one of its reads races the publisher's
/// write by construction. The engine waits on the word's own bytes and hands
/// back what it held on the iteration that satisfied `accepts`, so the checker
/// sees the word's write history and one verdict on it (API §3).
///
/// `accepts` runs here, where the thread-local scalars it reads still exist;
/// only a position in the write history crosses into the checker.
#[allow(private_bounds)]
#[inline(never)]
pub async fn declared_wait<T, S, Sem, W>(
    warp: &mut W,
    context: ExecCtx,
    site: SiteId,
    address: Address<S>,
    accepts: impl FnMut(u64, usize) -> Result<bool, crate::EngineError>,
) -> Result<R<T::Scalar>, super::EngineError>
where
    T: MemoryType,
    S: SpaceVariant,
    Sem: LoadSemantics,
    W: WarpHandle,
{
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(
        context,
        site,
        std::any::type_name::<(T, S, Sem)>(),
    );
    let warp = engine(warp);
    let context = context.into_inner();
    let (pointer, _logical_buffer) = address.into_parts();
    let pointer = pointer.with_byte_storage_access_width(T::Storage::BYTE_LEN)?;
    let mask = context.active_mask();
    let operation =
        warp.begin_optional_operation(context, site.get(), OperationKind::Load, false)?;
    let observed = warp
        .declared_word_wait_until(
            operation.as_ref(),
            &pointer,
            mask,
            T::Storage::BYTE_LEN,
            Sem::VALUE,
            accepts,
        )
        .await?;
    warp.finish_optional_operation(&operation)?;
    // The wait works in the word's bits; the destination wants the type the
    // kernel named. `decode_le` is the same door the ordinary load uses.
    let storage = R::<T::Storage>::from_fn(|lane| {
        let bits = observed[lane].to_le_bytes();
        T::Storage::decode_le(&bits[..T::Storage::BYTE_LEN]).unwrap_or_else(|_| T::Storage::zero())
    });
    Ok(T::decode_warp(storage))
}

#[inline(never)]
fn execute_address_load<T, S, W>(
    warp: &mut W,
    context: ExecCtx,
    site: SiteId,
    address: Address<S>,
    semantics: MemoryAccessSemantics,
    atomic_coherent: bool,
    readonly_proxy: bool,
) -> Result<R<T::Scalar>, super::EngineError>
where
    T: MemoryType,
    S: SpaceVariant,
    W: WarpHandle,
{
    let warp = engine(warp);
    let context = context.into_inner();
    let (pointer, logical_buffer) = address.into_parts();
    let pointer = pointer.with_byte_storage_access_width(T::Storage::BYTE_LEN)?;
    execute_physical_load::<T, W>(
        warp,
        context,
        site,
        pointer,
        logical_buffer,
        S::PTX_SPACE,
        semantics,
        atomic_coherent,
        readonly_proxy,
    )
}

#[inline(never)]
fn execute_direct_load<T, S, W>(
    warp: &mut W,
    context: ExecCtx,
    site: SiteId,
    address: DirectAddress<'_, S>,
    semantics: MemoryAccessSemantics,
    atomic_coherent: bool,
    readonly_proxy: bool,
) -> Result<R<T::Scalar>, super::EngineError>
where
    T: MemoryType,
    S: SpaceVariant,
    W: WarpHandle,
{
    if address.itemsize != T::Storage::BYTE_LEN {
        return Err(super::EngineError::message(format!(
            "ld access width {} does not match direct buffer itemsize {}",
            T::Storage::BYTE_LEN,
            address.itemsize,
        )));
    }
    let warp = engine(warp);
    let context = context.into_inner();
    let mask = context.active_mask();
    if !atomic_coherent && !readonly_proxy {
        let result = warp.execute_named_scalar_load::<T::Storage>(
            context,
            site.get(),
            address.logical_buffer,
            address.buffer,
            address.indices,
            mask,
        )?;
        return Ok(T::decode_warp(R::from_inner(result)));
    }
    let pointer = address
        .physical_pointer()
        .with_byte_storage_access_width(T::Storage::BYTE_LEN)?;
    execute_physical_load::<T, W>(
        warp,
        context,
        site,
        pointer,
        Some(std::sync::Arc::<str>::from(address.logical_buffer)),
        S::PTX_SPACE,
        semantics,
        atomic_coherent,
        readonly_proxy,
    )
}

#[inline(never)]
fn execute_physical_load<T, W>(
    warp: &mut crate::WarpEngine<W::Mode>,
    context: crate::WarpContext,
    site: SiteId,
    pointer: crate::runtime::PhysicalPtr,
    logical_buffer: Option<std::sync::Arc<str>>,
    ptx_space: PtxStateSpace,
    semantics: MemoryAccessSemantics,
    atomic_coherent: bool,
    readonly_proxy: bool,
) -> Result<R<T::Scalar>, super::EngineError>
where
    T: MemoryType,
    W: WarpHandle,
{
    let mask = context.active_mask();
    let same_space = first_generic_space_mask(&pointer, ptx_space, mask)?;
    if same_space != mask {
        let remaining = mask - same_space;
        let mut result = execute_physical_load::<T, W>(
            warp,
            context.with_active_mask(same_space),
            site,
            pointer.clone(),
            logical_buffer.clone(),
            ptx_space,
            semantics,
            atomic_coherent,
            readonly_proxy,
        )?
        .into_inner();
        let rest = execute_physical_load::<T, W>(
            warp,
            context.with_active_mask(remaining),
            site,
            pointer,
            logical_buffer,
            ptx_space,
            semantics,
            atomic_coherent,
            readonly_proxy,
        )?;
        result.masked_assign(remaining, rest.inner());
        return Ok(R::from_inner(result));
    }
    if readonly_proxy {
        pointer.observe_readonly_proxy(warp.kernel().physical(), mask, T::Storage::BYTE_LEN)?;
    }
    // NumSim executes generated control flow in program order and its numeric
    // memory already provides coherent warp-batched reads.  PTX ordering and
    // atomic-linearization metadata is consumed only by the checker modes.
    // Do not route numerical volatile/acquire loads through OrderingHub's
    // launch-wide reservation protocol.
    if !atomic_coherent || !warp.observes_operations() {
        let elided = warp.try_run_elided_named_physical_operation(
            context,
            site.get(),
            OperationKind::Load,
            logical_buffer.as_deref().unwrap_or("physical_pointer"),
            pointer.buffer(),
            |physical| {
                raw_load_physical_ptr_warp::<T::Storage>(
                    physical, &context, &pointer, mask, ptx_space,
                )
            },
        );
        if let Some(result) = elided {
            return Ok(T::decode_warp(R::from_inner(result?)));
        }
    }
    let operation =
        warp.begin_optional_pointer_operation(context, site.get(), OperationKind::Load, &pointer)?;
    let physical = warp.kernel().physical();
    let numeric = || {
        if atomic_coherent {
            raw_load_physical_ptr_warp_atomic::<T::Storage>(
                physical, &context, &pointer, mask, ptx_space,
            )
        } else {
            raw_load_physical_ptr_warp::<T::Storage>(physical, &context, &pointer, mask, ptx_space)
        }
    };
    let result = warp.physical_pointer_access(
        operation.as_ref(),
        OperationKind::Load,
        &pointer,
        logical_buffer.as_deref(),
        mask,
        T::Storage::BYTE_LEN,
        false,
        atomic_coherent,
        semantics,
        numeric,
    )?;
    warp.finish_optional_operation(&operation)?;
    Ok(T::decode_warp(R::from_inner(result)))
}

impl<T, S, Sem> ld_spec::sealed::Sealed for variant::Ld<T, S, Sem>
where
    T: MemoryType,
    S: SpaceVariant,
    Sem: LoadSemantics,
{
}

impl<T, S, Sem> ld_spec::Variant for variant::Ld<T, S, Sem>
where
    T: MemoryType,
    S: SpaceVariant,
    Sem: LoadSemantics,
{
    type Output = R<T::Scalar>;
}

#[inline(never)]
fn execute_address_store<T, S, W>(
    warp: &mut W,
    context: ExecCtx,
    site: SiteId,
    address: Address<S>,
    values: R<T::Scalar>,
    semantics: MemoryAccessSemantics,
) -> Result<(), super::EngineError>
where
    T: MemoryType,
    S: SpaceVariant,
    W: WarpHandle,
{
    let warp = engine(warp);
    let context = context.into_inner();
    let (pointer, logical_buffer) = address.into_parts();
    let pointer = pointer.with_byte_storage_access_width(T::Storage::BYTE_LEN)?;
    execute_physical_store::<T, W>(
        warp,
        context,
        site,
        pointer,
        logical_buffer,
        values,
        S::PTX_SPACE,
        semantics,
    )
}

#[inline(never)]
fn execute_direct_store<T, S, W>(
    warp: &mut W,
    context: ExecCtx,
    site: SiteId,
    address: DirectAddress<'_, S>,
    values: R<T::Scalar>,
    semantics: MemoryAccessSemantics,
    direct_named: bool,
) -> Result<(), super::EngineError>
where
    T: MemoryType,
    S: SpaceVariant,
    W: WarpHandle,
{
    if address.itemsize != T::Storage::BYTE_LEN {
        return Err(super::EngineError::message(format!(
            "st access width {} does not match direct buffer itemsize {}",
            T::Storage::BYTE_LEN,
            address.itemsize,
        )));
    }
    let warp = engine(warp);
    let context = context.into_inner();
    let mask = context.active_mask();
    if direct_named {
        let encoded = T::encode_warp(values);
        warp.execute_named_scalar_store::<T::Storage>(
            context,
            site.get(),
            address.logical_buffer,
            address.buffer,
            address.indices,
            encoded.inner(),
            mask,
        )?;
        return Ok(());
    }
    let pointer = address
        .physical_pointer()
        .with_byte_storage_access_width(T::Storage::BYTE_LEN)?;
    execute_physical_store::<T, W>(
        warp,
        context,
        site,
        pointer,
        Some(std::sync::Arc::<str>::from(address.logical_buffer)),
        values,
        S::PTX_SPACE,
        semantics,
    )
}

#[inline(never)]
fn execute_physical_store<T, W>(
    warp: &mut crate::WarpEngine<W::Mode>,
    context: crate::WarpContext,
    site: SiteId,
    pointer: crate::runtime::PhysicalPtr,
    logical_buffer: Option<std::sync::Arc<str>>,
    values: R<T::Scalar>,
    ptx_space: PtxStateSpace,
    semantics: MemoryAccessSemantics,
) -> Result<(), super::EngineError>
where
    T: MemoryType,
    W: WarpHandle,
{
    let mask = context.active_mask();
    let same_space = first_generic_space_mask(&pointer, ptx_space, mask)?;
    if same_space != mask {
        execute_physical_store::<T, W>(
            warp,
            context.with_active_mask(same_space),
            site,
            pointer.clone(),
            logical_buffer.clone(),
            values.clone(),
            ptx_space,
            semantics,
        )?;
        return execute_physical_store::<T, W>(
            warp,
            context.with_active_mask(mask - same_space),
            site,
            pointer,
            logical_buffer,
            values,
            ptx_space,
            semantics,
        );
    }
    let encoded = T::encode_warp(values);
    let elided = warp.try_run_elided_named_physical_operation(
        context,
        site.get(),
        OperationKind::Store,
        logical_buffer.as_deref().unwrap_or("physical_pointer"),
        pointer.buffer(),
        |physical| {
            raw_store_physical_ptr_warp::<T::Storage>(
                physical,
                &context,
                &pointer,
                encoded.inner(),
                mask,
                ptx_space,
            )
        },
    );
    if let Some(result) = elided {
        return result.map_err(Into::into);
    }
    let operation =
        warp.begin_optional_pointer_operation(context, site.get(), OperationKind::Store, &pointer)?;
    let physical = warp.kernel().physical();
    warp.physical_pointer_access(
        operation.as_ref(),
        OperationKind::Store,
        &pointer,
        logical_buffer.as_deref(),
        mask,
        T::Storage::BYTE_LEN,
        false,
        false,
        semantics,
        || {
            raw_store_physical_ptr_warp::<T::Storage>(
                physical,
                &context,
                &pointer,
                encoded.inner(),
                mask,
                ptx_space,
            )
        },
    )?;
    warp.finish_optional_operation(&operation)?;
    Ok(())
}

impl<T, S, Sem> st_spec::sealed::Sealed for variant::St<T, S, Sem>
where
    T: MemoryType,
    S: SpaceVariant,
    Sem: StoreSemantics,
{
}

impl<T, S, Sem> st_spec::Variant for variant::St<T, S, Sem>
where
    T: MemoryType,
    S: SpaceVariant,
    Sem: StoreSemantics,
{
    type Output = ();
}

// Rust has no stable `extern template`. These impls are the equivalent
// explicit-instantiation boundary: each simulator-observable semantic class,
// engine mode, and operand carrier gets one concrete entry in this crate.
// Exact PTX spellings that differ only in an unmodeled cache qualifier share
// that entry through inline trait glue.
//
// Each `*_for_mode!` below is one *row body*: it holds the impl text and
// nothing else. Which rows exist is decided further down by `mem_axis!` over
// the declarative carrier/space/semantics tables, and by `for_each_engine_mode!`
// over the single engine-mode list in `super::mode_axis`.
macro_rules! scalar_ld_for_mode {
    ($warp:ty, [$ty:ty, $space:ty, $sem:ty $(,)?]) => {
        impl ld_spec::sealed::Execute<$warp, Address<$space>> for variant::Ld<$ty, $space, $sem> {
            #[inline(never)]
            fn execute(
                warp: &mut $warp,
                context: ExecCtx,
                site: SiteId,
                address: Address<$space>,
            ) -> Result<Self::Output, super::EngineError> {
                execute_address_load::<$ty, $space, $warp>(
                    warp,
                    context,
                    site,
                    address,
                    <$sem as LoadSemantics>::VALUE,
                    <$sem as LoadSemantics>::ATOMIC_COHERENT,
                    <$sem as LoadSemantics>::READONLY_PROXY,
                )
            }
        }

        impl<'a> ld_spec::sealed::Execute<$warp, DirectAddress<'a, $space>>
            for variant::Ld<$ty, $space, $sem>
        {
            #[inline(never)]
            fn execute(
                warp: &mut $warp,
                context: ExecCtx,
                site: SiteId,
                address: DirectAddress<'a, $space>,
            ) -> Result<Self::Output, super::EngineError> {
                execute_direct_load::<$ty, $space, $warp>(
                    warp,
                    context,
                    site,
                    address,
                    <$sem as LoadSemantics>::VALUE,
                    <$sem as LoadSemantics>::ATOMIC_COHERENT,
                    <$sem as LoadSemantics>::READONLY_PROXY,
                )
            }
        }
    };
}

macro_rules! scalar_st_for_mode {
    ($warp:ty, [$ty:ty, $space:ty, $sem:ty $(,)?]) => {
        impl st_spec::sealed::Execute<$warp, (Address<$space>, R<<$ty as MemoryType>::Scalar>)>
            for variant::St<$ty, $space, $sem>
        {
            #[inline(never)]
            fn execute(
                warp: &mut $warp,
                context: ExecCtx,
                site: SiteId,
                (address, values): (Address<$space>, R<<$ty as MemoryType>::Scalar>),
            ) -> Result<Self::Output, super::EngineError> {
                execute_address_store::<$ty, $space, $warp>(
                    warp,
                    context,
                    site,
                    address,
                    values,
                    <$sem as StoreSemantics>::VALUE,
                )
            }
        }

        impl<'a>
            st_spec::sealed::Execute<
                $warp,
                (DirectAddress<'a, $space>, R<<$ty as MemoryType>::Scalar>),
            > for variant::St<$ty, $space, $sem>
        {
            #[inline(never)]
            fn execute(
                warp: &mut $warp,
                context: ExecCtx,
                site: SiteId,
                (address, values): (DirectAddress<'a, $space>, R<<$ty as MemoryType>::Scalar>),
            ) -> Result<Self::Output, super::EngineError> {
                execute_direct_store::<$ty, $space, $warp>(
                    warp,
                    context,
                    site,
                    address,
                    values,
                    <$sem as StoreSemantics>::VALUE,
                    <$sem as StoreSemantics>::DIRECT_NAMED,
                )
            }
        }
    };
}

// ---------------------------------------------------------------------------
// The instantiation lattice, as data.
//
// `mem_axis!` is the single cartesian-product driver over the
// `(carrier x state space)` axes; the carrier classes and the modeled state
// spaces below are the *only* copies of those lists in this file. Every
// instantiation row further down names a carrier class, a space class, an
// engine-mode gate, and the tail of the marker (semantics/cache or vector
// width). Adding a scalar carrier or a state space is one token in one list;
// adding an engine mode is one row in `super::mode_axis`.
//
// Rows are expanded by recursive munching rather than nested `$(...)` groups
// because two independently-sized lists cannot share one repetition depth.
macro_rules! mem_axis {
    ($gate:ident, $entry:ident, [$($class:ident),+ $(,)?], $spaces:tt, $extra:tt) => {
        $( mem_axis!(@class $gate, $entry, $class, $spaces, $extra); )+
    };

    // --- carrier axis -------------------------------------------------
    // `packed`: every scalar carrier the frontend can spell for a plain
    // lane-wise access, including the sub-word and packed-vector forms that
    // decode through `MemoryType`.
    (@class $gate:ident, $entry:ident, packed, $spaces:tt, $extra:tt) => {
        mem_axis!(@types $gate, $entry, [
            super::reg::variant::I8,
            super::reg::variant::I16,
            super::reg::variant::U8,
            super::reg::variant::U16,
            super::reg::variant::B32,
            super::reg::variant::B64,
            super::reg::variant::F16,
            super::reg::variant::Bf16,
            variant::Bool,
            variant::F16,
            variant::Bf16,
            variant::F16x2,
            variant::Bf16x2,
            variant::F32x2,
            variant::F32x4,
            variant::U64x2,
        ], $spaces, $extra);
    };
    // Raw pointer PTX forms are closed over the scalar carriers the frontend
    // validates for the b32/u32/s32/f32 and b64/u64/s64/f64 spellings. Adding
    // a raw scalar form therefore requires a token here instead of silently
    // creating a new downstream monomorphization.
    (@class $gate:ident, $entry:ident, w32, $spaces:tt, $extra:tt) => {
        mem_axis!(@types $gate, $entry, [
            super::reg::variant::I32,
            super::reg::variant::U32,
            super::reg::variant::F32,
        ], $spaces, $extra);
    };
    (@class $gate:ident, $entry:ident, w64, $spaces:tt, $extra:tt) => {
        mem_axis!(@types $gate, $entry, [
            super::reg::variant::I64,
            super::reg::variant::U64,
            super::reg::variant::F64,
        ], $spaces, $extra);
    };
    (@class $gate:ident, $entry:ident, w128, $spaces:tt, $extra:tt) => {
        mem_axis!(@types $gate, $entry, [variant::U64x2], $spaces, $extra);
    };
    // PTX's sub-word `.type` suffixes. `.b8`/`.u8`/`.s8` move one byte but
    // name a 32-bit register, so they need the widening carriers above.
    // `.b16`/`.u16`/`.s16` may name either a 16-bit register (class `w16`,
    // the identity carriers `packed` already instantiates for the plain
    // forms) or a wider 32-bit one (class `w16as32`), which PTX zero- or
    // sign-extends on load and truncates on store.
    (@class $gate:ident, $entry:ident, w8, $spaces:tt, $extra:tt) => {
        mem_axis!(@types $gate, $entry, [
            variant::U8AsU32,
            variant::S8AsI32,
        ], $spaces, $extra);
    };
    (@class $gate:ident, $entry:ident, w16, $spaces:tt, $extra:tt) => {
        mem_axis!(@types $gate, $entry, [
            super::reg::variant::U16,
            super::reg::variant::I16,
        ], $spaces, $extra);
    };
    (@class $gate:ident, $entry:ident, w16as32, $spaces:tt, $extra:tt) => {
        mem_axis!(@types $gate, $entry, [
            variant::U16AsU32,
            variant::S16AsI32,
        ], $spaces, $extra);
    };

    (@types $gate:ident, $entry:ident, [$(,)?], $spaces:tt, $extra:tt) => {};
    (@types $gate:ident, $entry:ident, [$ty:ty $(, $rest:ty)* $(,)?], $spaces:tt, $extra:tt) => {
        mem_axis!(@space_class $gate, $entry, $ty, $spaces, $extra);
        mem_axis!(@types $gate, $entry, [$($rest),*], $spaces, $extra);
    };

    // --- state-space axis ---------------------------------------------
    (@space_class $gate:ident, $entry:ident, $ty:ty, modeled, $extra:tt) => {
        mem_axis!(@spaces $gate, $entry, $ty,
            [Generic, Global, Shared, SharedCta, SharedCluster, Local], $extra);
    };
    (@space_class $gate:ident, $entry:ident, $ty:ty, [$($space:ty),+ $(,)?], $extra:tt) => {
        mem_axis!(@spaces $gate, $entry, $ty, [$($space),+], $extra);
    };

    (@spaces $gate:ident, $entry:ident, $ty:ty, [$(,)?], $extra:tt) => {};
    (@spaces $gate:ident, $entry:ident, $ty:ty, [$space:ty $(, $rest:ty)* $(,)?], [$($extra:tt)*]) => {
        for_each_engine_mode!($gate, $entry, [$ty, $space, $($extra)*]);
        mem_axis!(@spaces $gate, $entry, $ty, [$($rest),*], [$($extra)*]);
    };
}

/// The memory-scope axis of the scoped `.relaxed`/`.acquire`/`.release`
/// spellings. This is the only copy of the scope list on the instantiation
/// side; `scopes!` above is the matching value table.
macro_rules! mem_order_scopes {
    ($order:ident, $gate:ident, $entry:ident, $carriers:tt, $spaces:tt, [$($tail:tt)*]) => {
        mem_axis!($gate, $entry, $carriers, $spaces,
            [variant::$order<variant::Cta>, $($tail)*]);
        mem_axis!($gate, $entry, $carriers, $spaces,
            [variant::$order<variant::Cluster>, $($tail)*]);
        mem_axis!($gate, $entry, $carriers, $spaces,
            [variant::$order<variant::Gpu>, $($tail)*]);
        mem_axis!($gate, $entry, $carriers, $spaces,
            [variant::$order<variant::Sys>, $($tail)*]);
    };
}

// --- lane-wise `ld` rows ---------------------------------------------------
mem_axis!(
    test_visible,
    scalar_ld_for_mode,
    [w8, w16, w32, w64],
    [Global],
    [variant::Readonly]
);
mem_axis!(@types test_visible, scalar_ld_for_mode, [variant::F16, variant::Bf16], [Global], [variant::Readonly]);
mem_axis!(
    test_visible,
    scalar_ld_for_mode,
    [packed],
    modeled,
    [variant::Plain]
);
mem_axis!(
    analysis_only,
    scalar_ld_for_mode,
    [w32, w64],
    modeled,
    [variant::Plain]
);
mem_axis!(
    test_visible,
    scalar_ld_for_mode,
    [w32, w64, w128],
    modeled,
    [variant::Volatile]
);
mem_order_scopes!(
    Relaxed,
    test_visible,
    scalar_ld_for_mode,
    [w32, w64],
    modeled,
    []
);
mem_order_scopes!(
    Acquire,
    test_visible,
    scalar_ld_for_mode,
    [w32, w64],
    modeled,
    []
);
mem_axis!(
    test_visible,
    scalar_ld_for_mode,
    [w32, w64],
    [Global],
    [variant::Mmio]
);
mem_axis!(
    test_visible,
    scalar_ld_for_mode,
    [w32, w64, w128],
    [Global],
    [variant::MmioRelaxed]
);
mem_axis!(
    test_visible,
    scalar_ld_for_mode,
    [w32, w64],
    [Global],
    [variant::MmioAcquire]
);

// --- sub-word `ld` rows ----------------------------------------------------
// `w16` deliberately has no `Plain` row: the identity 16-bit carriers are
// already instantiated for the plain forms by the `packed` class above, and a
// second row for the same marker triple would be a conflicting impl. The
// widening `w16as32` carriers belong to no other class, so they do take one.
mem_axis!(
    test_visible,
    scalar_ld_for_mode,
    [w8, w16as32],
    modeled,
    [variant::Plain]
);
mem_axis!(
    test_visible,
    scalar_ld_for_mode,
    [w8, w16, w16as32],
    modeled,
    [variant::Volatile]
);
mem_order_scopes!(
    Relaxed,
    test_visible,
    scalar_ld_for_mode,
    [w8, w16, w16as32],
    modeled,
    []
);
mem_order_scopes!(
    Acquire,
    test_visible,
    scalar_ld_for_mode,
    [w8, w16, w16as32],
    modeled,
    []
);
mem_axis!(
    test_visible,
    scalar_ld_for_mode,
    [w8, w16, w16as32],
    [Global],
    [variant::Mmio]
);
mem_axis!(
    test_visible,
    scalar_ld_for_mode,
    [w8, w16, w16as32],
    [Global],
    [variant::MmioRelaxed]
);
mem_axis!(
    test_visible,
    scalar_ld_for_mode,
    [w8, w16, w16as32],
    [Global],
    [variant::MmioAcquire]
);

// --- lane-wise `st` rows ---------------------------------------------------
mem_axis!(
    test_visible,
    scalar_st_for_mode,
    [packed],
    modeled,
    [variant::Plain]
);
mem_axis!(
    analysis_only,
    scalar_st_for_mode,
    [w32, w64],
    modeled,
    [variant::Plain]
);
mem_axis!(
    test_visible,
    scalar_st_for_mode,
    [w32, w64],
    modeled,
    [variant::Volatile]
);
mem_order_scopes!(
    Relaxed,
    test_visible,
    scalar_st_for_mode,
    [w32, w64],
    modeled,
    []
);
mem_order_scopes!(
    Release,
    test_visible,
    scalar_st_for_mode,
    [w32, w64],
    modeled,
    []
);
mem_axis!(
    test_visible,
    scalar_st_for_mode,
    [w32, w64],
    [Global],
    [variant::Mmio]
);
mem_axis!(
    test_visible,
    scalar_st_for_mode,
    [w32, w64],
    [Global],
    [variant::MmioRelaxed]
);
mem_axis!(
    test_visible,
    scalar_st_for_mode,
    [w32, w64],
    [Global],
    [variant::MmioRelease]
);

// --- sub-word `st` rows ----------------------------------------------------
// Same shape as the sub-word `ld` rows, and `w16` again skips `Plain` because
// `packed` owns it while `w16as32` does not.
mem_axis!(
    test_visible,
    scalar_st_for_mode,
    [w8, w16as32],
    modeled,
    [variant::Plain]
);
mem_axis!(
    test_visible,
    scalar_st_for_mode,
    [w8, w16, w16as32],
    modeled,
    [variant::Volatile]
);
mem_order_scopes!(
    Relaxed,
    test_visible,
    scalar_st_for_mode,
    [w8, w16, w16as32],
    modeled,
    []
);
mem_order_scopes!(
    Release,
    test_visible,
    scalar_st_for_mode,
    [w8, w16, w16as32],
    modeled,
    []
);
mem_axis!(
    test_visible,
    scalar_st_for_mode,
    [w8, w16, w16as32],
    [Global],
    [variant::Mmio]
);
mem_axis!(
    test_visible,
    scalar_st_for_mode,
    [w8, w16, w16as32],
    [Global],
    [variant::MmioRelaxed]
);
mem_axis!(
    test_visible,
    scalar_st_for_mode,
    [w8, w16, w16as32],
    [Global],
    [variant::MmioRelease]
);

#[inline(never)]
fn execute_vector_store<T, S, const N: usize, const SINKS: u8, W>(
    warp: &mut W,
    context: ExecCtx,
    site: SiteId,
    address: Address<S>,
    values: [R<T::Scalar>; N],
    semantics: MemoryAccessSemantics,
) -> Result<(), super::EngineError>
where
    T: MemoryType,
    S: SpaceVariant,
    W: WarpHandle,
{
    let context = context.into_inner();
    let mask = context.active_mask();
    let same_space = first_generic_space_mask(address.inner(), S::PTX_SPACE, mask)?;
    if same_space != mask {
        for lanes in [same_space, mask - same_space] {
            execute_vector_store::<T, S, N, SINKS, W>(
                warp,
                ExecCtx::from_inner(context.with_active_mask(lanes)),
                site,
                address.clone(),
                values.clone(),
                semantics,
            )?;
        }
        return Ok(());
    }
    let warp = engine(warp);
    let byte_width = T::Storage::BYTE_LEN
        .checked_mul(N)
        .ok_or_else(|| crate::EngineError::message("vector store byte width overflow"))?;
    if SINKS != 0
        && (byte_width != 32
            || !matches!(S::PTX_SPACE, PtxStateSpace::Global | PtxStateSpace::Generic)
            || u16::from(SINKS) >= (1_u16 << N) - 1)
    {
        return Err(EngineError::message(
            "store sinks require a 256-bit global vector with at least one real source",
        )
        .into());
    }
    if mask.is_empty() {
        return Ok(());
    }
    let (pointer, logical_buffer) = address.into_parts();
    let pointer = pointer.with_byte_storage_access_width(byte_width)?;
    if byte_width == 32 {
        pointer.require_ptx_space_for_mask(PtxStateSpace::Global, mask)?;
    }
    let operation =
        warp.begin_optional_operation(context, site.get(), OperationKind::Store, false)?;
    let physical = warp.kernel().physical().clone();
    let numeric_effect = || {
        let encoded = values
            .iter()
            .cloned()
            .map(T::encode_warp)
            .collect::<Vec<_>>();
        let encoded = encoded.iter().map(R::inner).collect::<Vec<_>>();
        raw_store_vector_physical_ptr_warp::<T::Storage>(
            &physical,
            &context,
            &pointer,
            &encoded,
            mask,
            S::PTX_SPACE,
            SINKS,
        )
    };
    if SINKS == 0 && !semantics.order().is_strong() {
        warp.physical_pointer_access(
            operation.as_ref(),
            OperationKind::Store,
            &pointer,
            logical_buffer.as_deref(),
            mask,
            byte_width,
            false,
            false,
            semantics,
            numeric_effect,
        )?;
    } else {
        // Strong vector operations synchronize per scalar element, with or
        // without sinks. Keep them in one batch/event: separate instructions
        // would impose an order between elements that PTX does not provide.
        let space = match pointer.pointer_space_for_mask(mask)? {
            PointerSpace::Global => crate::PhysicalAccessSpace::Global,
            PointerSpace::Shared => crate::PhysicalAccessSpace::Shared,
            PointerSpace::Local => crate::PhysicalAccessSpace::Local,
            PointerSpace::Register => crate::PhysicalAccessSpace::Register,
        };
        warp.resolved_physical_access_batch(
            operation.as_ref(),
            crate::PhysicalAccessKind::Write,
            space,
            mask,
            |operation| {
                let descriptor = crate::PhysicalAccessDescriptor::new(
                    crate::PhysicalAccessKind::Write,
                    space,
                    (N - SINKS.count_ones() as usize) * T::Storage::BYTE_LEN,
                )
                .map_err(|error| EngineError::message(error.to_string()))?
                .with_memory_semantics(semantics);
                let mut batch = crate::PhysicalAccessBatch::resolve_unmerged(
                    operation.clone(),
                    descriptor,
                    |provenance| {
                        let lane = provenance.lane();
                        let base = pointer.lane_write_byte_offset(lane, byte_width)?;
                        (0..N)
                            .filter(|index| SINKS & (1 << index) == 0)
                            .map(|index| {
                                crate::runtime::resolve_runtime_physical_access(
                                    &context,
                                    pointer.buffer(),
                                    lane,
                                    base + index * T::Storage::BYTE_LEN,
                                    T::Storage::BYTE_LEN,
                                    crate::PhysicalAccessKind::Write,
                                )
                                .map(|access| access.span())
                            })
                            .collect::<Result<Vec<_>, EngineError>>()
                    },
                )
                .map_err(|error| EngineError::message(error.to_string()))?;
                if let Some(logical_buffer) = logical_buffer.as_deref() {
                    batch = batch.with_logical_buffer(logical_buffer);
                }
                Ok(batch)
            },
            numeric_effect,
        )?;
    }
    warp.finish_optional_operation(&operation)?;
    Ok(())
}

macro_rules! vector_store_variant {
    ($width:literal) => {
        impl<T, S, Sem, const SINKS: u8> st_spec::sealed::Sealed
            for variant::StVec<T, S, $width, Sem, SINKS>
        where
            T: MemoryType,
            S: SpaceVariant,
            Sem: StoreSemantics,
        {
        }

        impl<T, S, Sem, const SINKS: u8> st_spec::Variant
            for variant::StVec<T, S, $width, Sem, SINKS>
        where
            T: MemoryType,
            S: SpaceVariant,
            Sem: StoreSemantics,
        {
            type Output = ();
        }
    };
}

vector_store_variant!(2);
vector_store_variant!(4);
vector_store_variant!(8);

macro_rules! vector_st_for_mode {
    ($warp:ty, [$ty:ty, $space:ty, $sem:ty, $width:literal $(,)?]) => {
        impl<const SINKS: u8>
            st_spec::sealed::Execute<
                $warp,
                (Address<$space>, [R<<$ty as MemoryType>::Scalar>; $width]),
            > for variant::StVec<$ty, $space, $width, $sem, SINKS>
        {
            #[inline(never)]
            fn execute(
                warp: &mut $warp,
                context: ExecCtx,
                site: SiteId,
                (address, values): (Address<$space>, [R<<$ty as MemoryType>::Scalar>; $width]),
            ) -> Result<Self::Output, super::EngineError> {
                execute_vector_store::<$ty, $space, $width, SINKS, $warp>(
                    warp,
                    context,
                    site,
                    address,
                    values,
                    <$sem as StoreSemantics>::VALUE,
                )
            }
        }
    };
}

/// The store-side memory-order axis, shared by every vector width. Adding a
/// store ordering here covers all widths and spaces at once.
macro_rules! vector_store_forms {
    ($carriers:tt, $spaces:tt, $width:literal) => {
        mem_axis!(
            analysis_only,
            vector_st_for_mode,
            $carriers,
            $spaces,
            [variant::Plain, $width]
        );
        mem_axis!(
            analysis_only,
            vector_st_for_mode,
            $carriers,
            $spaces,
            [variant::Volatile, $width]
        );
        mem_order_scopes!(
            Relaxed,
            analysis_only,
            vector_st_for_mode,
            $carriers,
            $spaces,
            [$width]
        );
        mem_order_scopes!(
            Release,
            analysis_only,
            vector_st_for_mode,
            $carriers,
            $spaces,
            [$width]
        );
    };
}

// Accept the same vector forms as the frontend validator: v2 for 8/16/32/64
// bit types in every modeled space; v4 for 8/16/32-bit types everywhere and
// 64-bit types in global; v8 for 32-bit global stores only.
vector_store_forms!([w8, w16, w32, w64], modeled, 2);
vector_store_forms!([w8, w16, w32], modeled, 4);
vector_store_forms!([w64], [Global, Generic], 4);
vector_store_forms!([w32], [Global, Generic], 8);

trait AtomicType<Op>: MemoryType {
    const BYTE_LEN: usize;

    fn numeric(
        physical: &crate::PhysicalMemory,
        context: &crate::WarpContext,
        pointer: &crate::runtime::PhysicalPtr,
        operand: &crate::WarpValue<Self::Scalar>,
        mask: crate::WarpMask,
        ptx_space: PtxStateSpace,
    ) -> Result<crate::WarpValue<Self::Scalar>, crate::EngineError>;

    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        address: Address<Generic>,
        operand: R<Self::Scalar>,
        semantics: MemoryAccessSemantics,
        declared_space: PtxStateSpace,
        sync_relevant_return: bool,
    ) -> impl std::future::Future<Output = Result<R<Self::Scalar>, super::EngineError>> + Send;
}

macro_rules! scalar_atomic_type {
    ($type_marker:ty, $scalar:ty, $op_marker:ty, $operation:ident) => {
        impl AtomicType<$op_marker> for $type_marker {
            const BYTE_LEN: usize = <$scalar>::BYTE_LEN;

            fn numeric(
                physical: &crate::PhysicalMemory,
                context: &crate::WarpContext,
                pointer: &crate::runtime::PhysicalPtr,
                operand: &crate::WarpValue<$scalar>,
                mask: crate::WarpMask,
                ptx_space: PtxStateSpace,
            ) -> Result<crate::WarpValue<$scalar>, crate::EngineError> {
                raw_atomic_scalar_physical_ptr_warp(
                    physical,
                    context,
                    pointer,
                    operand,
                    mask,
                    ptx_space,
                    RawAtomicOperation::$operation,
                )
            }

            #[inline(never)]
            async fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                address: Address<Generic>,
                operand: R<$scalar>,
                semantics: MemoryAccessSemantics,
                declared_space: PtxStateSpace,
                sync_relevant_return: bool,
            ) -> Result<R<$scalar>, super::EngineError> {
                execute_atomic::<$type_marker, $op_marker>(
                    warp,
                    context,
                    site,
                    address,
                    operand,
                    semantics,
                    declared_space,
                    sync_relevant_return,
                )
                .await
            }
        }
    };
}

scalar_atomic_type!(super::reg::variant::I32, i32, variant::Add, Add);
scalar_atomic_type!(super::reg::variant::U32, u32, variant::Add, Add);
scalar_atomic_type!(super::reg::variant::U64, u64, variant::Add, Add);
scalar_atomic_type!(super::reg::variant::F32, f32, variant::Add, Add);
scalar_atomic_type!(super::reg::variant::F32, f32, variant::Add<true>, AddNoFtz);
scalar_atomic_type!(super::reg::variant::F64, f64, variant::Add, Add);
scalar_atomic_type!(super::reg::variant::U32, u32, variant::BitAnd, BitAnd);
scalar_atomic_type!(super::reg::variant::U64, u64, variant::BitAnd, BitAnd);
scalar_atomic_type!(super::reg::variant::U32, u32, variant::BitOr, BitOr);
scalar_atomic_type!(super::reg::variant::U64, u64, variant::BitOr, BitOr);
scalar_atomic_type!(super::reg::variant::U32, u32, variant::BitXor, BitXor);
scalar_atomic_type!(super::reg::variant::U64, u64, variant::BitXor, BitXor);
scalar_atomic_type!(super::reg::variant::U32, u32, variant::Exchange, Exchange);
scalar_atomic_type!(super::reg::variant::U64, u64, variant::Exchange, Exchange);
scalar_atomic_type!(
    variant::U64x2,
    crate::scalar::U64x2,
    variant::Exchange,
    Exchange
);
scalar_atomic_type!(super::reg::variant::U32, u32, variant::Increment, Increment);
scalar_atomic_type!(super::reg::variant::U32, u32, variant::Decrement, Decrement);
scalar_atomic_type!(super::reg::variant::I32, i32, variant::Minimum, Minimum);
scalar_atomic_type!(super::reg::variant::I64, i64, variant::Minimum, Minimum);
scalar_atomic_type!(super::reg::variant::U32, u32, variant::Minimum, Minimum);
scalar_atomic_type!(super::reg::variant::U64, u64, variant::Minimum, Minimum);
scalar_atomic_type!(super::reg::variant::I32, i32, variant::Maximum, Maximum);
scalar_atomic_type!(super::reg::variant::I64, i64, variant::Maximum, Maximum);
scalar_atomic_type!(super::reg::variant::U32, u32, variant::Maximum, Maximum);
scalar_atomic_type!(super::reg::variant::U64, u64, variant::Maximum, Maximum);

macro_rules! packed_atomic_add_type {
    ($marker:ty, $scalar:ty, $bytes:expr, $function:ident, with_space) => {
        impl AtomicType<variant::Add> for $marker {
            const BYTE_LEN: usize = $bytes;

            fn numeric(
                physical: &crate::PhysicalMemory,
                context: &crate::WarpContext,
                pointer: &crate::runtime::PhysicalPtr,
                operand: &crate::WarpValue<$scalar>,
                mask: crate::WarpMask,
                ptx_space: PtxStateSpace,
            ) -> Result<crate::WarpValue<$scalar>, crate::EngineError> {
                $function(physical, context, pointer, operand, mask, ptx_space)
            }

            #[inline(never)]
            async fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                address: Address<Generic>,
                operand: R<$scalar>,
                semantics: MemoryAccessSemantics,
                declared_space: PtxStateSpace,
                sync_relevant_return: bool,
            ) -> Result<R<$scalar>, super::EngineError> {
                execute_atomic::<$marker, variant::Add>(
                    warp,
                    context,
                    site,
                    address,
                    operand,
                    semantics,
                    declared_space,
                    sync_relevant_return,
                )
                .await
            }
        }
    };
    ($marker:ty, $scalar:ty, $bytes:expr, $function:ident, global_only) => {
        impl<const NOFTZ: bool> AtomicType<variant::Add<NOFTZ>> for $marker {
            const BYTE_LEN: usize = $bytes;

            fn numeric(
                physical: &crate::PhysicalMemory,
                context: &crate::WarpContext,
                pointer: &crate::runtime::PhysicalPtr,
                operand: &crate::WarpValue<$scalar>,
                mask: crate::WarpMask,
                ptx_space: PtxStateSpace,
            ) -> Result<crate::WarpValue<$scalar>, crate::EngineError> {
                if ptx_space != PtxStateSpace::Global {
                    return Err(crate::EngineError::message(
                        "this packed atomic.add variant requires global memory",
                    ));
                }
                $function::<NOFTZ>(physical, context, pointer, operand, mask)
            }

            #[inline(never)]
            async fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                address: Address<Generic>,
                operand: R<$scalar>,
                semantics: MemoryAccessSemantics,
                declared_space: PtxStateSpace,
                sync_relevant_return: bool,
            ) -> Result<R<$scalar>, super::EngineError> {
                execute_atomic::<$marker, variant::Add<NOFTZ>>(
                    warp,
                    context,
                    site,
                    address,
                    operand,
                    semantics,
                    declared_space,
                    sync_relevant_return,
                )
                .await
            }
        }
    };
}

macro_rules! half_vector_atomic_type {
    ($format:ty, $bf16:literal, $operation:ident) => {
        impl<const N: usize> AtomicType<variant::$operation> for variant::HalfVector<$format, N>
        where
            Self: MemoryType,
        {
            const BYTE_LEN: usize = N * 2;

            fn numeric(
                physical: &crate::PhysicalMemory,
                context: &crate::WarpContext,
                pointer: &crate::runtime::PhysicalPtr,
                operand: &crate::WarpValue<Self::Scalar>,
                mask: crate::WarpMask,
                ptx_space: PtxStateSpace,
            ) -> Result<crate::WarpValue<Self::Scalar>, EngineError> {
                raw_atomic_half_vector_physical_ptr_warp::<Self::Scalar, $bf16>(
                    physical,
                    context,
                    pointer,
                    operand,
                    mask,
                    ptx_space,
                    RawAtomicOperation::$operation,
                )
            }

            async fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                address: Address<Generic>,
                operand: R<Self::Scalar>,
                semantics: MemoryAccessSemantics,
                declared_space: PtxStateSpace,
                sync_relevant_return: bool,
            ) -> Result<R<Self::Scalar>, super::EngineError> {
                execute_atomic::<Self, variant::$operation>(
                    warp,
                    context,
                    site,
                    address,
                    operand,
                    semantics,
                    declared_space,
                    sync_relevant_return,
                )
                .await
            }
        }
    };
}

half_vector_atomic_type!(variant::F16, false, Add);
half_vector_atomic_type!(variant::F16, false, Minimum);
half_vector_atomic_type!(variant::F16, false, Maximum);
half_vector_atomic_type!(variant::Bf16, true, Add);
half_vector_atomic_type!(variant::Bf16, true, Minimum);
half_vector_atomic_type!(variant::Bf16, true, Maximum);

packed_atomic_add_type!(
    variant::F16,
    f32,
    2,
    raw_atomic_add_fp16_physical_ptr_warp,
    with_space
);
packed_atomic_add_type!(
    variant::Bf16,
    f32,
    2,
    raw_atomic_add_bf16_physical_ptr_warp,
    with_space
);
packed_atomic_add_type!(
    variant::F16x2,
    u32,
    4,
    raw_atomic_add_fp16x2_physical_ptr_warp,
    with_space
);
packed_atomic_add_type!(
    variant::Bf16x2,
    u32,
    4,
    raw_atomic_add_bf16x2_physical_ptr_warp,
    with_space
);
packed_atomic_add_type!(
    variant::F32x2,
    u64,
    8,
    raw_atomic_add_f32x2_physical_ptr_warp,
    global_only
);
packed_atomic_add_type!(
    variant::F32x4,
    crate::F32x4,
    16,
    raw_atomic_add_f32x4_physical_ptr_warp,
    global_only
);

#[inline(never)]
async fn execute_atomic<T, Op>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    address: Address<Generic>,
    operand: R<T::Scalar>,
    semantics: MemoryAccessSemantics,
    declared_space: PtxStateSpace,
    sync_relevant_return: bool,
) -> Result<R<T::Scalar>, super::EngineError>
where
    T: AtomicType<Op>,
{
    execute_atomic_access(
        warp,
        context,
        site,
        address,
        T::BYTE_LEN,
        semantics,
        declared_space,
        sync_relevant_return,
        |physical, context, pointer, mask, ptx_space| {
            T::numeric(physical, context, pointer, operand.inner(), mask, ptx_space)
        },
    )
    .await
}

/// Returning atomics, reductions, and CAS share one physical lifecycle.
/// Only the numeric callback and whether the return affects sync differ.
#[inline(never)]
async fn execute_atomic_access<T: RuntimeScalar + Send + Sync>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    address: Address<Generic>,
    byte_width: usize,
    semantics: MemoryAccessSemantics,
    declared_space: PtxStateSpace,
    sync_relevant_return: bool,
    numeric: impl Fn(
            &crate::PhysicalMemory,
            &crate::WarpContext,
            &crate::runtime::PhysicalPtr,
            crate::WarpMask,
            PtxStateSpace,
        ) -> Result<crate::WarpValue<T>, EngineError>
        + Send
        + Sync,
) -> Result<R<T>, super::EngineError> {
    let warp = engine(warp);
    let context = context.into_inner();
    let mut remaining = context.active_mask();
    let (pointer, logical_buffer) = address.into_parts();
    let pointer = pointer.with_byte_storage_access_width(byte_width)?;
    let physical = warp.kernel().physical().clone();
    let mut result = crate::WarpValue::splat(T::zero());
    while !remaining.is_empty() {
        let mask = first_generic_space_mask(&pointer, declared_space, remaining)?;
        let context = context.with_active_mask(mask);
        let ptx_space = resolve_atomic_space(declared_space, &pointer, mask)?;
        let operation =
            warp.begin_optional_operation(context, site.get(), OperationKind::Atomic, false)?;
        let values = warp
            .physical_pointer_atomic_access(
                operation.as_ref(),
                &pointer,
                logical_buffer.as_deref(),
                mask,
                byte_width,
                sync_relevant_return,
                semantics,
                || numeric(&physical, &context, &pointer, mask, ptx_space),
            )
            .await?;
        warp.finish_optional_operation(&operation)?;
        result.masked_assign(mask, &values);
        remaining = remaining - mask;
    }
    Ok(R::from_inner(result))
}

fn resolve_atomic_space(
    declared_space: PtxStateSpace,
    pointer: &crate::runtime::PhysicalPtr,
    mask: crate::WarpMask,
) -> Result<PtxStateSpace, crate::EngineError> {
    if declared_space != PtxStateSpace::Generic {
        return Ok(declared_space);
    }
    match pointer.pointer_space_for_mask(mask)? {
        PointerSpace::Global => Ok(PtxStateSpace::Global),
        PointerSpace::Shared => Ok(PtxStateSpace::Shared),
        _ => Err(crate::EngineError::message(
            "generic atom/red address must resolve to global or shared memory",
        )),
    }
}

instruction_variant! {
    [impl<T, S, Op, Sem>] atom_spec, variant::Atom<T, S, Op, Sem>
    where [
        T: AtomicType<Op>,
        S: AtomicSpace,
        Sem: AtomicSemantics,
    ],
    (Address<S>, R<T::Scalar>) => R<T::Scalar>;
    async fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (address, operand): Self::Args,
    ) -> Result<Self::Output, super::EngineError> {
        T::execute(
            warp,
            context,
            site,
            address.cast_space(),
            operand,
            Sem::VALUE,
            S::PTX_SPACE,
            true,
        )
        .await
    }
}

instruction_variant! {
    [impl<T, S, Op, Sem>] red_spec, variant::Red<T, S, Op, Sem>
    where [
        T: AtomicType<Op>,
        S: AtomicSpace,
        Sem: ReductionSemantics,
    ],
    (Address<S>, R<T::Scalar>) => ();
    async fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (address, operand): Self::Args,
    ) -> Result<Self::Output, super::EngineError> {
        let _ = T::execute(
            warp,
            context,
            site,
            address.cast_space(),
            operand,
            Sem::VALUE.as_reduction(),
            S::PTX_SPACE,
            false,
        )
        .await?;
        Ok(())
    }
}

instruction_variant! {
    [impl<const KIND: u8, const TYPE: u8, const OP: u8, const VEC: u8, const SEM: u8, const SCOPE: u8>]
    multimem_spec, variant::Multimem<KIND, TYPE, OP, VEC, SEM, SCOPE>,
    (Address<Global>, R<[u32; 4]>) => R<[u32; 4]>;
    async fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (address, operand): Self::Args,
    ) -> Result<Self::Output, super::EngineError> {
        let form = multimem_impl::MultimemForm::decode(KIND, TYPE, OP, VEC, SEM, SCOPE)?;
        multimem_impl::execute(warp, context, site, form, address, operand).await
    }
}

trait CasType: MemoryType {
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        address: Address<Generic>,
        compare: R<Self::Scalar>,
        replacement: R<Self::Scalar>,
        semantics: MemoryAccessSemantics,
        declared_space: PtxStateSpace,
    ) -> impl std::future::Future<Output = Result<R<Self::Scalar>, super::EngineError>> + Send;
}

macro_rules! cas_type {
    ($marker:ty) => {
        impl CasType for $marker {
            #[inline(never)]
            async fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                address: Address<Generic>,
                compare: R<Self::Scalar>,
                replacement: R<Self::Scalar>,
                semantics: MemoryAccessSemantics,
                declared_space: PtxStateSpace,
            ) -> Result<R<Self::Scalar>, super::EngineError> {
                execute_cas::<$marker>(
                    warp,
                    context,
                    site,
                    address,
                    compare,
                    replacement,
                    semantics,
                    declared_space,
                )
                .await
            }
        }
    };
}

cas_type!(super::reg::variant::I32);
cas_type!(super::reg::variant::I64);
cas_type!(super::reg::variant::U16);
cas_type!(super::reg::variant::U32);
cas_type!(super::reg::variant::U64);
cas_type!(super::reg::variant::B32);
cas_type!(super::reg::variant::B64);
cas_type!(variant::F32x4);
cas_type!(variant::U64x2);

#[inline(never)]
async fn execute_cas<T: CasType>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    address: Address<Generic>,
    compare: R<T::Scalar>,
    replacement: R<T::Scalar>,
    semantics: MemoryAccessSemantics,
    declared_space: PtxStateSpace,
) -> Result<R<T::Scalar>, super::EngineError> {
    execute_atomic_access(
        warp,
        context,
        site,
        address,
        T::Scalar::BYTE_LEN,
        semantics,
        declared_space,
        true,
        |physical, context, pointer, mask, ptx_space| {
            raw_atomic_cas_physical_ptr_warp::<T::Scalar>(
                physical,
                context,
                pointer,
                compare.inner(),
                replacement.inner(),
                mask,
                ptx_space,
            )
        },
    )
    .await
}

instruction_variant! {
    [impl<T, S, Sem>] atom_spec, variant::AtomCas<T, S, Sem>
    where [
        T: CasType,
        S: AtomicSpace,
        Sem: AtomicSemantics,
    ],
    (Address<S>, R<T::Scalar>, R<T::Scalar>) => R<T::Scalar>;
    async fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (address, compare, replacement): Self::Args,
    ) -> Result<Self::Output, super::EngineError> {
        T::execute(
            warp,
            context,
            site,
            address.cast_space(),
            compare,
            replacement,
            Sem::VALUE,
            S::PTX_SPACE,
        )
        .await
    }
}

#[inline(never)]
fn execute_ldmatrix<const COUNT: usize>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    source: Address<Shared>,
    transpose: bool,
    source_bits: usize,
) -> Result<R<[u32; COUNT]>, super::EngineError> {
    let warp = engine(warp);
    let context = context.into_inner();
    let mask = context.active_mask();
    let source = source.into_inner();
    let operation =
        warp.begin_optional_operation(context, site.get(), OperationKind::Load, false)?;
    let physical = warp.kernel().physical().clone();
    let fragments = warp.ldmatrix_access(
        operation.as_ref(),
        &context,
        &source,
        COUNT,
        transpose,
        source_bits,
        mask,
        || {
            if source_bits != 16 {
                return raw_ldmatrix_b8_fragments(
                    &physical,
                    &context,
                    &source,
                    COUNT,
                    transpose,
                    source_bits,
                );
            }
            raw_ldmatrix_b16_fragments(
                &physical,
                &context,
                &source,
                &crate::WarpValue::splat(0),
                1,
                COUNT,
                transpose,
                mask,
            )
        },
    )?;
    warp.finish_optional_operation(&operation)?;
    let fragments: [crate::WarpValue<u32>; COUNT] =
        fragments.try_into().map_err(|fragments: Vec<_>| {
            crate::EngineError::message(format!(
                "ldmatrix returned {} fragments for x{COUNT}",
                fragments.len()
            ))
        })?;
    Ok(R::from_fn(|lane| {
        std::array::from_fn(|fragment| fragments[fragment][lane])
    }))
}

macro_rules! ldmatrix_variant {
    ($count:expr) => {
        impl<const TRANSPOSE: bool, const SOURCE_BITS: usize, const SIGNED: bool>
            ldmatrix_spec::sealed::Sealed
            for variant::Ldmatrix<$count, TRANSPOSE, SOURCE_BITS, SIGNED>
        {
        }

        impl<const TRANSPOSE: bool, const SOURCE_BITS: usize, const SIGNED: bool>
            ldmatrix_spec::Variant for variant::Ldmatrix<$count, TRANSPOSE, SOURCE_BITS, SIGNED>
        {
            type Args = Address<Shared>;
            type Output = R<[u32; $count]>;
        }

        const _: () = {
            #[inline(never)]
            fn entry(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                source: Address<Shared>,
                transpose: bool,
                source_bits: usize,
            ) -> Result<R<[u32; $count]>, super::EngineError> {
                execute_ldmatrix::<$count>(warp, context, site, source, transpose, source_bits)
            }

            impl<const TRANSPOSE: bool, const SOURCE_BITS: usize, const SIGNED: bool>
                ldmatrix_spec::sealed::Execute
                for variant::Ldmatrix<$count, TRANSPOSE, SOURCE_BITS, SIGNED>
            {
                fn execute(
                    warp: &mut super::Engine,
                    context: ExecCtx,
                    site: SiteId,
                    source: Self::Args,
                ) -> Result<Self::Output, super::EngineError> {
                    if SIGNED && (SOURCE_BITS != 4 || TRANSPOSE) {
                        return Err(super::EngineError::message(
                            "signed ldmatrix requires non-transposed s4",
                        ));
                    }
                    let loaded = entry(warp, context, site, source, TRANSPOSE, SOURCE_BITS)?;
                    if !SIGNED {
                        return Ok(loaded);
                    }
                    Ok(R::from_fn(|lane| {
                        loaded.inner()[lane].map(|word| {
                            u32::from_le_bytes(
                                word.to_le_bytes()
                                    .map(|byte| ((byte as i8) << 4 >> 4) as u8),
                            )
                        })
                    }))
                }
            }
        };
    };
}

ldmatrix_variant!(1);
ldmatrix_variant!(2);
ldmatrix_variant!(4);

#[inline(never)]
fn execute_stmatrix<S: MemorySpace, const COUNT: usize>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    destination: Address<S>,
    registers: R<[u32; COUNT]>,
    descriptor: StmatrixDescriptor,
) -> Result<(), super::EngineError> {
    let warp = engine(warp);
    let context = context.into_inner();
    let destination = destination.into_inner();
    let operation =
        warp.begin_optional_operation(context, site.get(), OperationKind::Store, false)?;
    let physical = warp.kernel().physical().clone();
    let sources = std::array::from_fn::<_, COUNT, _>(|fragment| {
        crate::WarpValue::from_fn(|lane| registers[lane][fragment])
    });
    warp.stmatrix_access(
        operation.as_ref(),
        &context,
        &destination,
        COUNT,
        descriptor,
        || {
            let source_refs = sources.iter().collect::<Vec<_>>();
            raw_stmatrix(&physical, &context, &destination, &source_refs, descriptor)
        },
    )?;
    warp.finish_optional_operation(&operation)?;
    Ok(())
}

macro_rules! stmatrix_variant {
    ($marker:ty, $space:ty, $count:expr, $descriptor:expr) => {
        instruction_variant! {
            [impl] stmatrix_spec, $marker,
            (Address<$space>, R<[u32; $count]>) => ();
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                (destination, registers): Self::Args,
            ) -> Result<Self::Output, super::EngineError> {
                execute_stmatrix(warp, context, site, destination, registers, $descriptor)
            }
        }
    };
}

/// `stmatrix` instantiation lattice: `(fragment count x shared space x PTX
/// shape)`. Each axis appears exactly once.
macro_rules! stmatrix_lattice {
    ($($count:expr),+ $(,)?) => {
        $(
            stmatrix_lattice!(@space Shared, PtxStateSpace::Shared, $count);
            stmatrix_lattice!(@space SharedCta, PtxStateSpace::SharedCta, $count);
        )+
    };

    (@space $space:ty, $ptx:expr, $count:expr) => {
        stmatrix_variant!(
            variant::StmatrixM8N8B16<$space, $count, false>,
            $space,
            $count,
            StmatrixDescriptor::M8n8B16 {
                transpose: false,
                space: $ptx,
            }
        );
        stmatrix_variant!(
            variant::StmatrixM8N8B16<$space, $count, true>,
            $space,
            $count,
            StmatrixDescriptor::M8n8B16 {
                transpose: true,
                space: $ptx,
            }
        );
        stmatrix_variant!(
            variant::StmatrixM16N8B8<$space, $count>,
            $space,
            $count,
            StmatrixDescriptor::M16n8B8Transposed { space: $ptx }
        );
    };
}

stmatrix_lattice!(1, 2, 4);

#[inline(never)]
fn execute_st_bulk_zero<S: SpaceVariant>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    destination: Address<S>,
    byte_count: R<i64>,
) -> Result<(), super::EngineError> {
    let warp = engine(warp);
    let context = context.into_inner();
    // Zero-byte stores have no memory effect. Choose the effect mask before
    // creating its operation so numeric execution and both checkers agree.
    // Nonzero issuers stay in one SIMT operation, without invented lane order.
    let mask = crate::WarpMask::from_lanes(
        context
            .active_mask()
            .into_iter()
            .filter(|&lane| byte_count.inner()[lane] != 0),
    )
    .expect("a subset of a valid warp mask contains only valid lanes");
    if mask.is_empty() {
        return Ok(());
    }
    let context = context.with_active_mask(mask);
    let destination = destination.into_inner();
    let operation =
        warp.begin_optional_operation(context, site.get(), OperationKind::Store, false)?;
    let physical = warp.kernel().physical().clone();
    warp.st_bulk_zero_access(
        operation.as_ref(),
        &context,
        &destination,
        byte_count.inner(),
        mask,
        S::PTX_SPACE,
        || {
            raw_st_bulk_zero(
                &physical,
                &context,
                &destination,
                byte_count.inner(),
                mask,
                S::PTX_SPACE,
            )
        },
    )?;
    warp.finish_optional_operation(&operation)?;
    Ok(())
}

macro_rules! st_bulk_zero_variant {
    ($marker:ty, $space:ty) => {
        instruction_variant! {
            [impl] st_bulk_spec, $marker,
            (Address<$space>, R<i64>) => ();
            fn execute(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                (destination, byte_count): Self::Args,
            ) -> Result<Self::Output, super::EngineError> {
                execute_st_bulk_zero(warp, context, site, destination, byte_count)
            }
        }
    };
}

/// Discard is a weak write of indeterminate bytes, not a cache-only no-op.
pub fn discard(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    address: Address<Global>,
) -> Result<(), EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(context, site, std::any::type_name_of_val(&discard));
    let warp = engine(warp);
    let context = context.into_inner();
    let mask = context.active_mask();
    let address = address.into_inner();
    address.require_ptx_space_for_mask(PtxStateSpace::Global, mask)?;
    for lane in mask {
        if address.lane_physical_byte_offset(lane, 128)? % 128 != 0 {
            return Err(EngineError::message(
                "discard requires a 128-byte aligned address",
            ));
        }
    }
    let operation =
        warp.begin_optional_operation(context, site.get(), OperationKind::Store, false)?;
    let physical = warp.kernel().physical().clone();
    warp.resolved_physical_access_batch(
        operation.as_ref(),
        crate::PhysicalAccessKind::Write,
        crate::PhysicalAccessSpace::Global,
        mask,
        |operation| {
            let descriptor = crate::PhysicalAccessDescriptor::new(
                crate::PhysicalAccessKind::Write,
                crate::PhysicalAccessSpace::Global,
                128,
            )
            .map_err(|error| EngineError::message(error.to_string()))?;
            crate::PhysicalAccessBatch::resolve_single_span(
                operation.clone(),
                descriptor,
                |provenance| {
                    let lane = provenance.lane();
                    crate::runtime::resolve_runtime_physical_access(
                        &context,
                        address.buffer(),
                        lane,
                        address.lane_write_byte_offset(lane, 128)?,
                        128,
                        crate::PhysicalAccessKind::Write,
                    )
                    .map(|access| access.span())
                },
            )
            .map_err(|error| EngineError::message(error.to_string()))
        },
        || {
            for lane in mask {
                crate::runtime::invalidate_runtime_bytes(
                    &physical,
                    &context,
                    address.buffer(),
                    lane,
                    address.lane_write_byte_offset(lane, 128)?,
                    128,
                )?;
            }
            Ok(())
        },
    )?;
    warp.finish_optional_operation(&operation)
}

st_bulk_zero_variant!(variant::StBulkZeroGeneric, Generic);
st_bulk_zero_variant!(variant::StBulkZeroShared, Shared);

#[cfg(all(test, not(feature = "analysis-core")))]
mod tests {
    use super::*;
    use crate::abi::v2::control::{branch_context, Ordinary};
    use crate::runtime::{
        run_kernel_engine_launch, ExecutionPolicy, LaunchSelection, PhysicalPtr, RuntimeBuffer,
    };
    use crate::{LaunchTopology, NumSimMode, PhysicalMemory, WarpValue};
    use std::sync::Arc;

    #[test]
    fn typed_load_and_store_share_one_observable_physical_effect_path() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let physical = PhysicalMemory::new(topology);
        let allocation = physical.global().allocate_zeroed(4).unwrap();
        let pointer = PhysicalPtr::new(
            RuntimeBuffer::Global(physical.global().full_view(allocation).unwrap()),
            WarpValue::splat(0),
            4,
        );
        run_kernel_engine_launch::<NumSimMode, _, _>(
            physical,
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let pointer = pointer.clone();
                async move {
                    let root = ExecCtx::from_inner(warp.context());
                    let lane_zero =
                        branch_context::<Ordinary>(root, super::super::LaneMask::from_bits(1))?;
                    let address = Address::<Global>::from_inner(pointer);
                    st::<variant::St<super::super::reg::variant::U32, Global>>(
                        &mut warp,
                        lane_zero,
                        SiteId::new(1),
                        (address.clone(), R::splat(0x1234_5678)),
                    )?;
                    let loaded = ld::<variant::Ld<super::super::reg::variant::U32, Global>>(
                        &mut warp,
                        lane_zero,
                        SiteId::new(2),
                        address,
                    )?;
                    assert_eq!(loaded[0], 0x1234_5678);
                    Ok(())
                }
            },
        )
        .unwrap();
    }

    /// PTX ISA 9.7.9.8: a sub-word load "is sign-extended to the destination
    /// register width for signed integers, and is zero-extended ... for
    /// unsigned and bit-size types", and the matching store keeps only the low
    /// byte. Both carriers must therefore touch exactly one byte while
    /// presenting a 32-bit register.
    #[test]
    fn sub_word_carriers_move_one_byte_and_extend_into_a_32_bit_register() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let physical = PhysicalMemory::new(topology);
        let allocation = physical.global().allocate_zeroed(4).unwrap();
        let pointer = PhysicalPtr::new(
            RuntimeBuffer::Global(physical.global().full_view(allocation).unwrap()),
            WarpValue::splat(0),
            1,
        );
        run_kernel_engine_launch::<NumSimMode, _, _>(
            physical,
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let pointer = pointer.clone();
                async move {
                    let root = ExecCtx::from_inner(warp.context());
                    let lane_zero =
                        branch_context::<Ordinary>(root, super::super::LaneMask::from_bits(1))?;
                    let address = Address::<Global>::from_inner(pointer);

                    // Only the low byte of the 32-bit register is stored.
                    st::<variant::St<variant::U8AsU32, Global>>(
                        &mut warp,
                        lane_zero,
                        SiteId::new(1),
                        (address.clone(), R::splat(0xdead_be80_u32)),
                    )?;
                    let zero_extended = ld::<variant::Ld<variant::U8AsU32, Global>>(
                        &mut warp,
                        lane_zero,
                        SiteId::new(2),
                        address.clone(),
                    )?;
                    assert_eq!(zero_extended[0], 0x80);

                    let sign_extended = ld::<variant::Ld<variant::S8AsI32, Global>>(
                        &mut warp,
                        lane_zero,
                        SiteId::new(3),
                        address.clone(),
                    )?;
                    assert_eq!(sign_extended[0], -128);

                    st::<variant::St<variant::S8AsI32, Global>>(
                        &mut warp,
                        lane_zero,
                        SiteId::new(4),
                        (address.clone(), R::splat(-1_i32)),
                    )?;
                    let all_ones = ld::<variant::Ld<variant::U8AsU32, Global>>(
                        &mut warp,
                        lane_zero,
                        SiteId::new(5),
                        address,
                    )?;
                    assert_eq!(all_ones[0], 0xff);
                    Ok(())
                }
            },
        )
        .unwrap();
    }

    #[test]
    fn half_word_carriers_move_two_bytes_and_extend_into_a_32_bit_register() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let physical = PhysicalMemory::new(topology);
        let allocation = physical.global().allocate_zeroed(4).unwrap();
        let pointer = PhysicalPtr::new(
            RuntimeBuffer::Global(physical.global().full_view(allocation).unwrap()),
            WarpValue::splat(0),
            2,
        );
        run_kernel_engine_launch::<NumSimMode, _, _>(
            physical,
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let pointer = pointer.clone();
                async move {
                    let root = ExecCtx::from_inner(warp.context());
                    let lane_zero =
                        branch_context::<Ordinary>(root, super::super::LaneMask::from_bits(1))?;
                    let address = Address::<Global>::from_inner(pointer);

                    // Only the low half of the 32-bit register is stored.
                    st::<variant::St<variant::U16AsU32, Global>>(
                        &mut warp,
                        lane_zero,
                        SiteId::new(1),
                        (address.clone(), R::splat(0xdead_8000_u32)),
                    )?;
                    let zero_extended = ld::<variant::Ld<variant::U16AsU32, Global>>(
                        &mut warp,
                        lane_zero,
                        SiteId::new(2),
                        address.clone(),
                    )?;
                    assert_eq!(zero_extended[0], 0x8000);

                    let sign_extended = ld::<variant::Ld<variant::S16AsI32, Global>>(
                        &mut warp,
                        lane_zero,
                        SiteId::new(3),
                        address.clone(),
                    )?;
                    assert_eq!(sign_extended[0], -32768);

                    st::<variant::St<variant::S16AsI32, Global>>(
                        &mut warp,
                        lane_zero,
                        SiteId::new(4),
                        (address.clone(), R::splat(-1_i32)),
                    )?;
                    let all_ones = ld::<variant::Ld<variant::U16AsU32, Global>>(
                        &mut warp,
                        lane_zero,
                        SiteId::new(5),
                        address,
                    )?;
                    assert_eq!(all_ones[0], 0xffff);
                    Ok(())
                }
            },
        )
        .unwrap();
    }

    #[test]
    fn atom_and_red_share_the_atomic_core_but_keep_distinct_return_contracts() {
        let topology = LaunchTopology::new(1, 1, 1).unwrap();
        let physical = PhysicalMemory::new(topology);
        let allocation = physical.global().allocate_zeroed(4).unwrap();
        let pointer = PhysicalPtr::new(
            RuntimeBuffer::Global(physical.global().full_view(allocation).unwrap()),
            WarpValue::splat(0),
            4,
        );
        run_kernel_engine_launch::<NumSimMode, _, _>(
            physical,
            0,
            Arc::new(()),
            LaunchSelection::default(),
            1,
            ExecutionPolicy::default(),
            move |mut warp| {
                let pointer = pointer.clone();
                async move {
                    type U32 = super::super::reg::variant::U32;
                    let root = ExecCtx::from_inner(warp.context());
                    let lane_zero =
                        branch_context::<Ordinary>(root, super::super::LaneMask::from_bits(1))?;
                    let address = Address::<Global>::from_inner(pointer);
                    st::<variant::St<U32, Global>>(
                        &mut warp,
                        lane_zero,
                        SiteId::new(10),
                        (address.clone(), R::splat(7)),
                    )?;
                    let previous = atom::<
                        variant::Atom<U32, Global, variant::Add, variant::Relaxed<variant::Gpu>>,
                    >(
                        &mut warp,
                        lane_zero,
                        SiteId::new(11),
                        (address.clone(), R::splat(5)),
                    )
                    .await?;
                    assert_eq!(previous[0], 7);
                    red::<
                        variant::Red<U32, Global, variant::BitXor, variant::Release<variant::Gpu>>,
                    >(
                        &mut warp,
                        lane_zero,
                        SiteId::new(12),
                        (address.clone(), R::splat(3)),
                    )
                    .await?;
                    let before_cas =
                        atom::<variant::AtomCas<U32, Global, variant::AcqRel<variant::Gpu>>>(
                            &mut warp,
                            lane_zero,
                            SiteId::new(13),
                            (address.clone(), R::splat(15), R::splat(99)),
                        )
                        .await?;
                    assert_eq!(before_cas[0], 15);
                    let loaded = ld::<variant::Ld<U32, Global>>(
                        &mut warp,
                        lane_zero,
                        SiteId::new(14),
                        address,
                    )?;
                    assert_eq!(loaded[0], 99);
                    Ok(())
                }
            },
        )
        .unwrap();
    }
}
