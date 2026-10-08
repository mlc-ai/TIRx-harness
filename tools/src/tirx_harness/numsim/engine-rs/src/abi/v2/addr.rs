//! Address-conversion PTX instructions.

use super::instruction::sync_instruction;
use super::{Address, EngineError, ExecCtx, Generic, Shared, SiteId, R};
use crate::runtime::abi_transport::PtxAddressSpace;

/// Query the state-space owner of a resolved generic address.
#[allow(private_bounds)]
pub fn isspacep<S: PtxAddressSpace>(
    _warp: &mut super::Engine,
    context: ExecCtx,
    _site: SiteId,
    address: Address<Generic>,
) -> Result<R<bool>, EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(context, _site, std::any::type_name_of_val(&isspacep::<S>));
    let mut result = crate::WarpValue::splat(false);
    for lane in context.active_mask().into_inner() {
        result[lane] = address.inner().is_in_ptx_address_space(
            &context.into_inner(),
            lane,
            S::PTX_SPACE,
        )?;
    }
    Ok(R::from_inner(result))
}

/// Both generic and shared spellings resolve the same physical CTA owner.
pub fn getctarank(
    _warp: &mut super::Engine,
    context: ExecCtx,
    _site: SiteId,
    address: Address<Shared>,
) -> Result<R<u32>, EngineError> {
    #[cfg(feature = "profile")]
    crate::instruction_profile::record(context, _site, std::any::type_name_of_val(&getctarank));
    let ranks = address
        .inner()
        .shared_target_cta_ranks(&context.into_inner(), context.active_mask().into_inner())?;
    Ok(R::from_inner(ranks.map(|_, rank| rank as u32)))
}

sync_instruction!(cvta_spec, CvtaVariant, cvta);

/// Static PTX spellings supported by the address instruction family.
pub mod variant {
    pub struct Convert<S, const TO_GENERIC: bool>(std::marker::PhantomData<S>);
    /// Observe a generic pointer in the selected CTA/cluster shared window.
    /// The u64 PTX spelling zero-extends the same shared32 address.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct GenericToSharedU32<const CLUSTER: bool>;

}

impl<S: PtxAddressSpace, const TO_GENERIC: bool> cvta_spec::sealed::Sealed
    for variant::Convert<S, TO_GENERIC>
{
}
impl<S: PtxAddressSpace, const TO_GENERIC: bool> cvta_spec::Variant
    for variant::Convert<S, TO_GENERIC>
{
    type Args = R<u64>;
    type Output = R<u64>;
}
impl<S: PtxAddressSpace, const TO_GENERIC: bool> cvta_spec::sealed::Execute
    for variant::Convert<S, TO_GENERIC>
{
    fn execute(
        warp: &mut super::Engine,
        context: ExecCtx,
        _site: SiteId,
        address: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        let bits = warp.kernel().physical().convert_ptx_address(
            &context.into_inner(),
            address.inner(),
            S::PTX_SPACE,
            TO_GENERIC,
        )?;
        Ok(R::from_inner(bits))
    }
}

impl<const CLUSTER: bool> cvta_spec::sealed::Sealed for variant::GenericToSharedU32<CLUSTER> {}

impl<const CLUSTER: bool> cvta_spec::Variant for variant::GenericToSharedU32<CLUSTER> {
    type Args = Address<Generic>;
    type Output = R<u32>;
}

impl<const CLUSTER: bool> cvta_spec::sealed::Execute for variant::GenericToSharedU32<CLUSTER> {
    fn execute(
        _warp: &mut super::Engine,
        context: ExecCtx,
        _site: SiteId,
        address: Self::Args,
    ) -> Result<Self::Output, EngineError> {
        // The generic marker is the PTX input state space. The physical
        // allocation retained by Address is the independent oracle that this
        // conversion really targets shared memory.
        address
            .inner()
            .shared_byte_addresses_u32(
                &context.into_inner(),
                context.active_mask().into_inner(),
            )
            .map(R::from_inner)
            .map_err(Into::into)
    }
}
