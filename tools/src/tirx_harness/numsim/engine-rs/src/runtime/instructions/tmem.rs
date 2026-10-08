//! Engine implementation of v2 source-level scalar TMEM accesses.
//!
//! PTX has no scalar `ld.tmem`/`st.tmem` instruction.  TIR nevertheless
//! permits a scalar `TensorLoad`/`BufferStore` on a TMEM view before the CUDA
//! backend has selected a concrete `tcgen05.ld`/`tcgen05.st` packing.  These
//! two calls preserve that precise source operation for frontends which run
//! before instruction selection.  Instruction-selected frontends must use
//! [`crate::abi::v2::tcgen05::ld`] and [`crate::abi::v2::tcgen05::st`]
//! instead.

use std::marker::PhantomData;

use super::instruction::{instruction_variant, sync_instruction};
use super::mem::MemoryType;
use super::transport::engine;
use super::{BufferHandle, EngineError, ExecCtx, SiteId, Tmem, R};
use crate::runtime::{load_tmem_scalar_warp, store_tmem_scalar_warp};
use crate::{OperationKind, RuntimeScalar, TmemAccessMode};

sync_instruction!(read_spec, ReadVariant, read);
sync_instruction!(write_spec, WriteVariant, write);

/// Static source-level TMEM access forms.
pub mod variant {
    use super::PhantomData;

    /// Scalar storage type and static/dynamic TMEM allocation discipline.
    pub struct Access<T, Mode>(PhantomData<fn() -> (T, Mode)>);
}

trait AccessMode {
    const VALUE: TmemAccessMode;
}

impl AccessMode for super::tcgen05::variant::StaticTmem {
    const VALUE: TmemAccessMode = TmemAccessMode::Static;
}

impl AccessMode for super::tcgen05::variant::DynamicTmem {
    const VALUE: TmemAccessMode = TmemAccessMode::Dynamic;
}

type ReadArgs = (BufferHandle<Tmem>, R<i64>, R<i64>, R<i64>);
type WriteArgs<T> = (BufferHandle<Tmem>, R<i64>, R<i64>, R<i64>, R<T>);

trait TmemEntry: MemoryType {
    fn read(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: ReadArgs,
        mode: TmemAccessMode,
    ) -> Result<R<Self::Scalar>, EngineError>;

    fn write(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        args: WriteArgs<Self::Scalar>,
        mode: TmemAccessMode,
    ) -> Result<(), EngineError>;
}

macro_rules! tmem_entry {
    ($marker:ty) => {
        impl TmemEntry for $marker {
            #[inline(never)]
            fn read(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                args: ReadArgs,
                mode: TmemAccessMode,
            ) -> Result<R<Self::Scalar>, EngineError> {
                execute_read::<$marker>(warp, context, site, args, mode)
            }

            #[inline(never)]
            fn write(
                warp: &mut super::Engine,
                context: ExecCtx,
                site: SiteId,
                args: WriteArgs<Self::Scalar>,
                mode: TmemAccessMode,
            ) -> Result<(), EngineError> {
                execute_write::<$marker>(warp, context, site, args, mode)
            }
        }
    };
}

macro_rules! tmem_entries {
    ($($marker:ty),+ $(,)?) => {
        $(tmem_entry!($marker);)+
    };
}

tmem_entries!(
    super::reg::variant::I8,
    super::reg::variant::I16,
    super::reg::variant::I32,
    super::reg::variant::I64,
    super::reg::variant::U8,
    super::reg::variant::U16,
    super::reg::variant::U32,
    super::reg::variant::U64,
    super::reg::variant::B32,
    super::reg::variant::B64,
    super::reg::variant::F16,
    super::reg::variant::Bf16,
    super::reg::variant::F32,
    super::reg::variant::F64,
    super::mem::variant::Bool,
    super::mem::variant::F16,
    super::mem::variant::Bf16,
    super::mem::variant::F16x2,
    super::mem::variant::Bf16x2,
    super::mem::variant::F32x2,
    super::mem::variant::F32x4,
    super::mem::variant::U64x2,
);

instruction_variant! {
    [impl<T, Mode>] read_spec, variant::Access<T, Mode>
    where [
        T: TmemEntry,
        Mode: AccessMode,
    ],
    ReadArgs => R<T::Scalar>;
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (buffer, mapped_lanes, columns, allocated_addresses): Self::Args,
    ) -> Result<Self::Output, EngineError> {
        T::read(
            warp,
            context,
            site,
            (buffer, mapped_lanes, columns, allocated_addresses),
            Mode::VALUE,
        )
    }
}

#[inline(never)]
fn execute_read<T: MemoryType>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    (buffer, mapped_lanes, columns, allocated_addresses): ReadArgs,
    mode: TmemAccessMode,
) -> Result<R<T::Scalar>, EngineError> {
    let warp = engine(warp);
    let context = context.into_inner();
    let mask = context.active_mask();
    let operation =
        warp.begin_optional_operation(context, site.get(), OperationKind::Load, false)?;
    let physical = warp.kernel().physical().clone();
    let lifecycle = warp.kernel().services().tcgen();
    let logical_buffer = buffer.logical_buffer().unwrap_or("tmem").to_owned();
    let buffer = buffer.inner().clone();
    let mapped_lanes = mapped_lanes.into_inner();
    let columns = columns.into_inner();
    let allocated_addresses = allocated_addresses.into_inner();
    let values = warp.runtime_named_tmem_physical_access(
        operation.as_ref(),
        OperationKind::Load,
        T::Storage::BYTE_LEN,
        &logical_buffer,
        mode,
        &buffer,
        &mapped_lanes,
        &columns,
        &allocated_addresses,
        mask,
        false,
        || {
            load_tmem_scalar_warp::<T::Storage>(
                &physical,
                &context,
                &lifecycle,
                mode,
                &buffer,
                &mapped_lanes,
                &columns,
                &allocated_addresses,
                mask,
            )
        },
    )?;
    warp.finish_optional_operation(&operation)?;
    Ok(R::from_inner(values).map(|_lane, value| T::decode(value)))
}

instruction_variant! {
    [impl<T, Mode>] write_spec, variant::Access<T, Mode>
    where [
        T: TmemEntry,
        Mode: AccessMode,
    ],
    WriteArgs<T::Scalar> => ();
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        site: SiteId,
        (buffer, mapped_lanes, columns, allocated_addresses, values): Self::Args,
    ) -> Result<Self::Output, EngineError> {
        T::write(
            warp,
            context,
            site,
            (buffer, mapped_lanes, columns, allocated_addresses, values),
            Mode::VALUE,
        )
    }
}

#[inline(never)]
fn execute_write<T: MemoryType>(
    warp: &mut super::Engine,
    context: ExecCtx,
    site: SiteId,
    (buffer, mapped_lanes, columns, allocated_addresses, values): WriteArgs<T::Scalar>,
    mode: TmemAccessMode,
) -> Result<(), EngineError> {
    let warp = engine(warp);
    let context = context.into_inner();
    let mask = context.active_mask();
    let operation =
        warp.begin_optional_operation(context, site.get(), OperationKind::Store, false)?;
    let physical = warp.kernel().physical().clone();
    let lifecycle = warp.kernel().services().tcgen();
    let logical_buffer = buffer.logical_buffer().unwrap_or("tmem").to_owned();
    let buffer = buffer.inner().clone();
    let mapped_lanes = mapped_lanes.into_inner();
    let columns = columns.into_inner();
    let allocated_addresses = allocated_addresses.into_inner();
    let encoded = values.map(|_lane, value| T::encode(value));
    warp.runtime_named_tmem_physical_access(
        operation.as_ref(),
        OperationKind::Store,
        T::Storage::BYTE_LEN,
        &logical_buffer,
        mode,
        &buffer,
        &mapped_lanes,
        &columns,
        &allocated_addresses,
        mask,
        false,
        || {
            store_tmem_scalar_warp::<T::Storage>(
                &physical,
                &context,
                &lifecycle,
                mode,
                &buffer,
                &mapped_lanes,
                &columns,
                &allocated_addresses,
                encoded.inner(),
                mask,
            )
        },
    )?;
    warp.finish_optional_operation(&operation)?;
    Ok(())
}
