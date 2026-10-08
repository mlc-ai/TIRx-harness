use std::sync::{Arc, Mutex};

use numsim_engine::abi::v2::{
    self as v2, Address, ElementMap, ElementRef, EngineError, ExecCtx, Global, LaneId,
    LogicalCoord, MapError, MappedView, Register, Shared, SiteId, Tmem, R,
};
#[cfg(not(feature = "analysis-core"))]
use numsim_engine::artifact_support::NumSimWarpEngine as TestWarpEngine;
#[cfg(feature = "racecheck")]
use numsim_engine::artifact_support::RaceCheckWarpEngine as TestWarpEngine;
#[cfg(all(feature = "analysis-core", not(feature = "racecheck")))]
use numsim_engine::artifact_support::SyncCheckWarpEngine as TestWarpEngine;
use numsim_engine::artifact_support::{
    run_kernel_launch_ordered, ExecutionPolicy, LaunchSelection, LaunchTopology, PhysicalMemory,
};

fn nymph_floor_div_mod(
    context: ExecCtx,
    lhs: R<i32>,
    rhs: R<i32>,
) -> Result<(R<i32>, R<i32>), EngineError> {
    use v2::reg::variant as rv;

    let zero = R::splat(0_i32);
    let one = R::splat(1_i32);
    let truncated = v2::reg::div::<rv::I32>(context, SiteId::new(100), (lhs.clone(), rhs.clone()))?;
    let product =
        v2::reg::mul::<rv::I32>(context, SiteId::new(101), (truncated.clone(), rhs.clone()))?;
    let remainder = v2::reg::sub::<rv::I32>(context, SiteId::new(102), (lhs.clone(), product))?;
    let nonzero = v2::reg::setp::<rv::Setp<rv::I32, rv::Ne>>(
        context,
        SiteId::new(103),
        (remainder.clone(), zero.clone()),
    )?;
    let remainder_negative = v2::reg::setp::<rv::Setp<rv::I32, rv::Lt>>(
        context,
        SiteId::new(104),
        (remainder, zero.clone()),
    )?;
    let divisor_negative = v2::reg::setp::<rv::Setp<rv::I32, rv::Lt>>(
        context,
        SiteId::new(105),
        (rhs.clone(), zero.clone()),
    )?;
    let divisor_negative_i32 = v2::reg::selp::<rv::I32>(
        context,
        SiteId::new(106),
        (divisor_negative.clone(), one.clone(), zero.clone()),
    )?;
    let divisor_nonnegative_i32 = v2::reg::selp::<rv::I32>(
        context,
        SiteId::new(107),
        (divisor_negative, zero.clone(), one),
    )?;
    let signs_differ = v2::reg::selp::<rv::I32>(
        context,
        SiteId::new(108),
        (
            remainder_negative,
            divisor_nonnegative_i32,
            divisor_negative_i32,
        ),
    )?;
    let adjustment =
        v2::reg::selp::<rv::I32>(context, SiteId::new(109), (nonzero, signs_differ, zero))?;
    let quotient = v2::reg::sub::<rv::I32>(context, SiteId::new(110), (truncated, adjustment))?;
    let floor_product =
        v2::reg::mul::<rv::I32>(context, SiteId::new(111), (quotient.clone(), rhs))?;
    let modulo = v2::reg::sub::<rv::I32>(context, SiteId::new(112), (lhs, floor_product))?;
    Ok((quotient, modulo))
}

#[cfg(not(feature = "analysis-core"))]
async fn execute_smoke(
    mut warp: TestWarpEngine,
    context: ExecCtx,
    observed: Arc<Mutex<Vec<i32>>>,
) -> Result<(), EngineError> {
    let lhs = R::from_fn(|lane| lane as i32);
    let rhs = R::splat(7_i32);
    let sum = v2::reg::add::<v2::reg::variant::I32>(context, SiteId::new(1), (lhs, rhs))?;

    let narrowed = v2::reg::cvt::<
        v2::reg::variant::Cvt<v2::reg::variant::F32, v2::reg::variant::F16, v2::reg::variant::Rn>,
    >(context, SiteId::new(3), R::splat(1.0_f32))?;
    let widened = v2::reg::cvt::<
        v2::reg::variant::Cvt<v2::reg::variant::F16, v2::reg::variant::F32>,
    >(context, SiteId::new(4), narrowed)?;
    let less = v2::reg::setp::<v2::reg::variant::Setp<v2::reg::variant::F32, v2::reg::variant::Lt>>(
        context,
        SiteId::new(5),
        (widened, R::splat(2.0_f32)),
    )?;
    assert!(less[0]);

    let f16_sum = v2::reg::add::<v2::reg::variant::F16Rn>(
        context,
        SiteId::new(6),
        (R::splat(0x3c00_u16), R::splat(0x3c00_u16)),
    )?;
    assert_eq!(f16_sum[0], 0x4000);
    let bf16_fma = v2::reg::fma::<v2::reg::variant::Bf16Rn>(
        context,
        SiteId::new(7),
        (
            R::splat(0x3fc0_u16),
            R::splat(0x4000_u16),
            R::splat(0x3f00_u16),
        ),
    )?;
    assert_eq!(bf16_fma[0], 0x4060);

    let quotient = v2::reg::div::<v2::reg::variant::I32>(
        context,
        SiteId::new(8),
        (R::splat(-7_i32), R::splat(3_i32)),
    )?;
    assert_eq!(quotient[0], -2); // v2's signed-integer div model.
    let floored = v2::reg::cvt::<
        v2::reg::variant::Cvt<v2::reg::variant::F32, v2::reg::variant::F32, v2::reg::variant::Rmi>,
    >(context, SiteId::new(9), R::splat(-2.25_f32))?;
    assert_eq!(floored[0], -3.0);
    let rounded_integer = v2::reg::cvt::<
        v2::reg::variant::Cvt<v2::reg::variant::I32, v2::reg::variant::F32, v2::reg::variant::Rn>,
    >(context, SiteId::new(10), R::splat(16_777_217_i32))?;
    assert_eq!(rounded_integer[0], 16_777_216.0);

    // The scalar PTX `cvt` grammar forms, one specialization per modifier set.
    let truncated = v2::reg::cvt::<
        v2::reg::variant::Cvt<
            v2::reg::variant::F32,
            v2::reg::variant::I32,
            v2::reg::variant::CvtMode<v2::reg::variant::Rzi, v2::reg::variant::PreserveSubnormal>,
        >,
    >(context, SiteId::new(200), R::splat(f32::NAN))?;
    assert_eq!(truncated[0], 0);
    let widened_nan = v2::reg::cvt::<
        v2::reg::variant::Cvt<
            v2::reg::variant::F64,
            v2::reg::variant::I64,
            v2::reg::variant::CvtMode<v2::reg::variant::Rmi>,
        >,
    >(context, SiteId::new(201), R::splat(f64::NAN))?;
    assert_eq!(widened_nan[0], i64::MIN);
    let toward_zero = v2::reg::cvt::<
        v2::reg::variant::Cvt<
            v2::reg::variant::U64,
            v2::reg::variant::F32,
            v2::reg::variant::CvtMode<v2::reg::variant::Rz, v2::reg::variant::PreserveSubnormal>,
        >,
    >(context, SiteId::new(202), R::splat(16_777_217_u64))?;
    assert_eq!(toward_zero[0], 16_777_216.0);
    let clamped = v2::reg::cvt::<
        v2::reg::variant::Cvt<
            v2::reg::variant::F32,
            v2::reg::variant::Bf16,
            v2::reg::variant::PackedMode<
                v2::reg::variant::Rn,
                v2::reg::variant::SatFinite,
                v2::reg::variant::NoRelu,
            >,
        >,
    >(context, SiteId::new(203), R::splat(f32::MAX))?;
    assert_eq!(clamped[0], 0x7f7f);
    let tf32 = v2::reg::cvt::<
        v2::reg::variant::Cvt<
            v2::reg::variant::F32,
            v2::reg::variant::Tf32,
            v2::reg::variant::PackedMode<v2::reg::variant::Rna>,
        >,
    >(
        context,
        SiteId::new(204),
        R::splat(f32::from_bits(0x3f80_1000)),
    )?;
    assert_eq!(tf32[0], 0x3f80_2000);

    let floor_lhs = R::from_fn(|lane| [-7, 7, -7, 6][lane % 4]);
    let floor_rhs = R::from_fn(|lane| [3, -3, -3, 3][lane % 4]);
    let (floor_quotient, floor_modulo) = nymph_floor_div_mod(context, floor_lhs, floor_rhs)?;
    assert_eq!(
        [
            floor_quotient[0],
            floor_quotient[1],
            floor_quotient[2],
            floor_quotient[3],
        ],
        [-3, -3, 2, 2],
    );
    assert_eq!(
        [
            floor_modulo[0],
            floor_modulo[1],
            floor_modulo[2],
            floor_modulo[3],
        ],
        [2, -2, -1, 0],
    );

    let child = v2::control::branch_context::<v2::control::Ordinary>(
        context,
        v2::LaneMask::from_bits(0x0f),
    )?;
    assert_eq!(child.active_mask().bits(), 0x0f);
    let inactive_zero_divisors = R::from_fn(|lane| if lane < 4 { 2_i32 } else { 0_i32 });
    let child_quotient = v2::reg::div::<v2::reg::variant::I32>(
        child,
        SiteId::new(11),
        (R::splat(8_i32), inactive_zero_divisors),
    )?;
    assert_eq!(child_quotient[0], 4);
    assert_eq!(child_quotient[4], 0);

    v2::control::for_enter(&mut warp, SiteId::new(2), 0)?;
    v2::control::for_exit(&mut warp, SiteId::new(2), 1, v2::LaneMask::EMPTY).await?;
    observed.lock().unwrap().push(sum[0]);
    Ok(())
}

#[cfg(not(feature = "analysis-core"))]
#[test]
fn external_artifact_uses_only_the_public_v2_contract() {
    let topology = LaunchTopology::new(1, 1, 1).unwrap();
    let physical = PhysicalMemory::new(topology);
    let observed = Arc::new(Mutex::new(Vec::new()));
    let callback_observed = Arc::clone(&observed);
    let stats = run_kernel_launch_ordered(
        &physical,
        0,
        LaunchSelection::default(),
        1,
        ExecutionPolicy::default(),
        move |mut warp, _services| {
            let context = v2::warp::context(&mut warp);
            let observed = Arc::clone(&callback_observed);
            async move {
                execute_smoke(warp, context, observed)
                    .await
                    .map_err(Into::into)
            }
        },
    )
    .unwrap();

    assert_eq!(stats.task_count, 1);
    assert_eq!(stats.completed_task_count, 1);
    assert_eq!(*observed.lock().unwrap(), vec![7]);
}

struct LinearMap;

impl ElementMap<Global> for LinearMap {
    type Runtime = usize;

    fn map(
        &self,
        logical: LogicalCoord<'_>,
        lane: LaneId,
        itemsize: &Self::Runtime,
    ) -> Result<ElementRef<Global>, MapError> {
        let [index] = logical.dimensions() else {
            return Err(MapError::new("linear map requires rank one"));
        };
        let byte_offset = i128::from(*index)
            .checked_mul(*itemsize as i128)
            .ok_or_else(|| MapError::new("linear offset overflow"))?;
        Ok(ElementRef::<Global>::in_bounds(
            byte_offset,
            Some(lane.index() as u32),
        ))
    }
}

type Shape32 = v2::tile::variant::Shape1<32>;
type TileCopy = v2::tile::variant::Copy<
    Shape32,
    v2::reg::variant::U32,
    Global,
    Shared,
    v2::tile::variant::Warp,
    v2::tile::variant::SnapshotSync,
    v2::tile::variant::NoFill,
>;
type Classic =
    v2::tile::variant::CpAsync<Shape32, v2::reg::variant::U32, v2::tile::variant::Warp, 4>;
type Bulk = v2::tile::variant::BulkS2g<Shape32, v2::reg::variant::U32, v2::tile::variant::Warp>;
type Tensor = v2::tile::variant::TensorS2g<Shape32, v2::reg::variant::U32, v2::tile::variant::Warp>;
type TensorReduce = v2::tile::variant::TensorS2gReduce<
    Shape32,
    v2::reg::variant::U32,
    v2::tile::variant::Warp,
    v2::tile::variant::ReduceAdd,
>;

type RawCp = v2::tcgen05::variant::Cp<
    v2::tcgen05::variant::Cp4x256b,
    v2::tcgen05::variant::NoDecompress,
    1,
    v2::tcgen05::variant::StaticTmem,
>;
type RemoteAsyncStore = v2::async_copy::variant::StAsyncClusterCompleteTxBytesU32x4;
type RawLd = v2::tcgen05::variant::Ld<
    v2::tcgen05::variant::Shape16x64b,
    v2::tcgen05::variant::Num<1>,
    false,
    v2::tcgen05::variant::StaticTmem,
>;
type RawSt = v2::tcgen05::variant::St<
    v2::tcgen05::variant::Shape16x64b,
    v2::tcgen05::variant::Num<1>,
    false,
    v2::tcgen05::variant::StaticTmem,
>;
type TileCp =
    v2::tile::variant::Tcgen05Cp<Shape32, v2::reg::variant::U32, v2::reg::variant::U32, RawCp>;
type TileLd =
    v2::tile::variant::Tcgen05Ld<Shape32, v2::reg::variant::U32, v2::reg::variant::U32, RawLd>;
type TileSt =
    v2::tile::variant::Tcgen05St<Shape32, v2::reg::variant::U32, v2::reg::variant::U32, RawSt>;

type DenseGemm = v2::tile::variant::Gemm<
    v2::tile::variant::Dense,
    v2::reg::variant::F16,
    v2::reg::variant::Bf16,
    v2::tile::variant::AShared,
    v2::tcgen05::variant::StaticTmem,
    16,
    8,
    16,
    16,
    8,
    16,
    1,
    false,
    false,
>;

type DenseWsGemm = v2::tile::variant::Gemm<
    v2::tile::variant::Dense,
    v2::reg::variant::F16,
    v2::reg::variant::Bf16,
    v2::tile::variant::AShared,
    v2::tcgen05::variant::StaticTmem,
    16,
    8,
    16,
    16,
    8,
    16,
    1,
    false,
    false,
>;

type DenseWsBatchedGemm = v2::tile::variant::Gemm<
    v2::tile::variant::Dense,
    v2::reg::variant::F16,
    v2::reg::variant::Bf16,
    v2::tile::variant::ATmem,
    v2::tcgen05::variant::StaticTmem,
    64,
    32,
    16,
    64,
    32,
    16,
    1,
    false,
    false,
    1,
    0,
    { u32::MAX },
    v2::tile::variant::MappedWsBatched,
>;

type DenseCta2BankedAGemm = v2::tile::variant::Gemm<
    v2::tile::variant::Dense,
    v2::reg::variant::F16,
    v2::reg::variant::Bf16,
    v2::tile::variant::ATmem,
    v2::tcgen05::variant::StaticTmem,
    64,
    32,
    16,
    128,
    32,
    16,
    2,
    false,
    false,
    1,
    0,
    { u32::MAX },
    v2::tile::variant::MappedCta2BankedA,
>;

type MixedBlockGemm = v2::tile::variant::Gemm<
    v2::tile::variant::BlockScaled<v2::tile::variant::E8m0>,
    v2::tile::variant::E2m1,
    v2::tile::variant::E4m3,
    v2::tile::variant::AShared,
    v2::tcgen05::variant::StaticTmem,
    128,
    128,
    64,
    256,
    128,
    64,
    2,
    false,
    false,
    16,
>;

fn assert_copy<V: v2::tile::CopyVariant<Source = Global, Destination = Shared>>() {}
fn assert_classic<V: v2::tile::CpAsyncVariant<Args = ()>>() {}
fn assert_raw_cp_async<V: v2::async_copy::CpAsyncVariant<Output = ()>>() {}
fn assert_raw_bulk_reduce<V>()
where
    V: v2::async_copy::CpReduceAsyncBulkVariant<
        Args = (Address<Global>, Address<Shared>, R<i64>),
        Output = (),
    >,
{
}
fn assert_remote_async_store<
    V: v2::async_copy::StAsyncVariant<
        Args = (Address<Shared>, Address<Shared>, R<i64>, [R<u32>; 4]),
        Output = (),
    >,
>() {
}
fn assert_bulk<V: v2::tile::BulkCopyVariant<Args = (), Source = Shared, Destination = Global>>() {}
fn assert_tensor<
    V: v2::tile::TensorCopyVariant<Args = (), Source = Shared, Destination = Global>,
>() {
}
fn assert_reduce<V: v2::tile::TensorReduceVariant<Args = ()>>() {}
fn assert_cp<V: v2::tile::Tcgen05CpVariant>() {}
fn assert_ld<V: v2::tile::Tcgen05LdVariant>() {}
fn assert_st<V: v2::tile::Tcgen05StVariant>() {}
fn assert_gemm<V: v2::tile::GemmAsyncVariant>() {}
fn assert_gemm_ws<V: v2::tile::GemmAsyncWsVariant>() {}
fn assert_add_u16<V: v2::reg::AddVariant<Args = (R<u16>, R<u16>), Output = R<u16>>>() {}
fn assert_sub_u16<V: v2::reg::SubVariant<Args = (R<u16>, R<u16>), Output = R<u16>>>() {}
fn assert_mul_u16<V: v2::reg::MulVariant<Args = (R<u16>, R<u16>), Output = R<u16>>>() {}
fn assert_mul_wide_i16<V: v2::reg::MulVariant<Args = (R<i16>, R<i16>), Output = R<i32>>>() {}
fn assert_mad_wide_u16<V: v2::reg::MadVariant<Args = (R<u16>, R<u16>, R<u32>), Output = R<u32>>>() {
}
fn assert_mad_f32<V: v2::reg::MadVariant<Args = (R<f32>, R<f32>, R<f32>), Output = R<f32>>>() {}
fn assert_mad_f64<V: v2::reg::MadVariant<Args = (R<f64>, R<f64>, R<f64>), Output = R<f64>>>() {}
fn assert_fma_u16<V: v2::reg::FmaVariant<Args = (R<u16>, R<u16>, R<u16>), Output = R<u16>>>() {}
fn assert_clmad_u64<V: v2::reg::ClmadVariant<Args = (R<u64>, R<u64>, R<u64>), Output = R<u64>>>() {}
fn assert_lop3<V: v2::reg::Lop3Variant<Args = (R<u32>, R<u32>, R<u32>), Output = R<u32>>>() {}
fn assert_lop3_bool<
    V: v2::reg::Lop3Variant<Args = (R<u32>, R<u32>, R<u32>, R<bool>), Output = (R<u32>, R<bool>)>,
>() {
}
fn assert_div_i32<V: v2::reg::DivVariant<Args = (R<i32>, R<i32>), Output = R<i32>>>() {}
fn assert_div_u32<V: v2::reg::DivVariant<Args = (R<u32>, R<u32>), Output = R<u32>>>() {}
fn assert_div_i64<V: v2::reg::DivVariant<Args = (R<i64>, R<i64>), Output = R<i64>>>() {}
fn assert_div_u64<V: v2::reg::DivVariant<Args = (R<u64>, R<u64>), Output = R<u64>>>() {}
fn assert_min_u16<V: v2::reg::MinVariant<Args = (R<u16>, R<u16>), Output = R<u16>>>() {}
fn assert_max_u16<V: v2::reg::MaxVariant<Args = (R<u16>, R<u16>), Output = R<u16>>>() {}
fn assert_min_u32<V: v2::reg::MinVariant<Args = (R<u32>, R<u32>), Output = R<u32>>>() {}
fn assert_max_u32<V: v2::reg::MaxVariant<Args = (R<u32>, R<u32>), Output = R<u32>>>() {}
fn assert_max_f32_three<
    V: v2::reg::MaxVariant<Args = (R<f32>, R<f32>, R<f32>), Output = R<f32>>,
>() {
}
fn assert_cvt_down<V: v2::reg::CvtVariant<Args = R<f32>, Output = R<u16>>>() {}
fn assert_cvt_up<V: v2::reg::CvtVariant<Args = R<u16>, Output = R<f32>>>() {}
fn assert_cvt_f32<V: v2::reg::CvtVariant<Args = R<f32>, Output = R<f32>>>() {}
fn assert_cvt_i32_f32<V: v2::reg::CvtVariant<Args = R<i32>, Output = R<f32>>>() {}
fn assert_cvt_u32_f32<V: v2::reg::CvtVariant<Args = R<u32>, Output = R<f32>>>() {}
fn assert_cvt_i64_f32<V: v2::reg::CvtVariant<Args = R<i64>, Output = R<f32>>>() {}
fn assert_cvt_u64_f32<V: v2::reg::CvtVariant<Args = R<u64>, Output = R<f32>>>() {}
fn assert_cvt_f32_pair_to_packed_byte<
    V: v2::reg::CvtVariant<Args = (R<f32>, R<f32>), Output = R<u16>>,
>() {
}
fn assert_cvt_packed_word_to_packed_byte<V: v2::reg::CvtVariant<Args = R<u32>, Output = R<u16>>>() {
}
fn assert_cvt_scaled_packed_word_to_packed_byte<
    V: v2::reg::CvtVariant<Args = (R<u32>, R<u8>), Output = R<u16>>,
>() {
}
fn assert_cvt_packed_byte_to_packed_word<V: v2::reg::CvtVariant<Args = R<u16>, Output = R<u32>>>() {
}
fn assert_cvt_to_packed_word<A, V: v2::reg::CvtVariant<Args = A, Output = R<u32>>>() {}
fn assert_cvt_f32_pair_to_nibble_pair<
    V: v2::reg::CvtVariant<Args = (R<f32>, R<f32>), Output = R<u8>>,
>() {
}
fn assert_cvt_packed_word_to_nibble_pair<V: v2::reg::CvtVariant<Args = R<u32>, Output = R<u8>>>() {}
fn assert_cvt_f32_quad_to_packed_word<
    V: v2::reg::CvtVariant<Args = (R<f32>, R<f32>, R<f32>, R<f32>, R<u32>), Output = R<u32>>,
>() {
}
fn assert_cvt_f32_quad_to_nibble_quad<
    V: v2::reg::CvtVariant<Args = (R<f32>, R<f32>, R<f32>, R<f32>, R<u32>), Output = R<u16>>,
>() {
}
fn assert_cvt_scaled_to_packed_word<
    V: v2::reg::CvtVariant<Args = (R<u16>, R<u16>), Output = R<u32>>,
>() {
}
fn assert_setp_f32<V: v2::reg::SetpVariant<Args = (R<f32>, R<f32>), Output = R<bool>>>() {}
fn assert_setp_u16<V: v2::reg::SetpVariant<Args = (R<u16>, R<u16>), Output = R<bool>>>() {}
fn assert_set_u32<V: v2::reg::SetVariant<Args = (R<f32>, R<f32>), Output = R<u32>>>() {}
fn assert_set_packed<V: v2::reg::SetVariant<Args = (R<u32>, R<u32>), Output = R<u32>>>() {}
fn assert_set_bool_u32<
    V: v2::reg::SetVariant<Args = (R<u32>, R<u32>, R<bool>), Output = R<u32>>,
>() {
}
fn assert_setp_bool<V: v2::reg::SetpVariant<Args = (R<u16>, R<u16>, R<bool>), Output = R<bool>>>() {
}
fn assert_setp_pair<
    V: v2::reg::SetpVariant<Args = (R<u32>, R<u32>), Output = (R<bool>, R<bool>)>,
>() {
}
fn assert_setp_pair_bool<
    V: v2::reg::SetpVariant<Args = (R<f32>, R<f32>, R<bool>), Output = (R<bool>, R<bool>)>,
>() {
}
fn assert_slct_u32<V: v2::reg::SlctVariant<Args = (R<u32>, R<u32>, R<f32>), Output = R<u32>>>() {}
fn assert_testp_f32<V: v2::reg::TestpVariant<Args = R<f32>, Output = R<bool>>>() {}
fn assert_add_i32<V: v2::reg::AddVariant<Args = (R<i32>, R<i32>), Output = R<i32>>>() {}
fn assert_sub_i32<V: v2::reg::SubVariant<Args = (R<i32>, R<i32>), Output = R<i32>>>() {}
fn assert_mul_i32<V: v2::reg::MulVariant<Args = (R<i32>, R<i32>), Output = R<i32>>>() {}
fn assert_min_i32<V: v2::reg::MinVariant<Args = (R<i32>, R<i32>), Output = R<i32>>>() {}
fn assert_max_i32<V: v2::reg::MaxVariant<Args = (R<i32>, R<i32>), Output = R<i32>>>() {}
fn assert_and_i32<V: v2::reg::AndVariant<Args = (R<i32>, R<i32>), Output = R<i32>>>() {}
fn assert_and_bool<V: v2::reg::AndVariant<Args = (R<bool>, R<bool>), Output = R<bool>>>() {}
fn assert_or_i32<V: v2::reg::OrVariant<Args = (R<i32>, R<i32>), Output = R<i32>>>() {}
fn assert_xor_i32<V: v2::reg::XorVariant<Args = (R<i32>, R<i32>), Output = R<i32>>>() {}
fn assert_shl_i32<V: v2::reg::ShlVariant<Args = (R<i32>, R<u32>), Output = R<i32>>>() {}
fn assert_shr_u32<V: v2::reg::ShrVariant<Args = (R<u32>, R<u32>), Output = R<u32>>>() {}
fn assert_shr_u64<V: v2::reg::ShrVariant<Args = (R<u64>, R<u32>), Output = R<u64>>>() {}
fn assert_neg_i32<V: v2::reg::NegVariant<Args = R<i32>, Output = R<i32>>>() {}
fn assert_selp_i32<V: v2::reg::SelpVariant<Args = (R<bool>, R<i32>, R<i32>), Output = R<i32>>>() {}
fn assert_selp_u16<V: v2::reg::SelpVariant<Args = (R<bool>, R<u16>, R<u16>), Output = R<u16>>>() {}
fn assert_mov_u32<V: v2::reg::MovVariant<Args = (), Output = R<u32>>>() {}
fn assert_mov_pack_b16x2<V: v2::reg::MovPackVariant<Args = (R<u16>, R<u16>), Output = R<u32>>>() {}
fn assert_mov_pack_b32x2<V: v2::reg::MovPackVariant<Args = (R<u32>, R<u32>), Output = R<u64>>>() {}
fn assert_mov_pack_b64x2<
    V: v2::reg::MovPackVariant<Args = (R<u64>, R<u64>), Output = R<[u64; 2]>>,
>() {
}
fn assert_mov_unpack_b16x2<
    V: v2::reg::MovUnpackVariant<Args = R<u32>, Output = (R<u16>, R<u16>)>,
>() {
}
fn assert_mov_unpack_b32x2<
    V: v2::reg::MovUnpackVariant<Args = R<u64>, Output = (R<u32>, R<u32>)>,
>() {
}
fn assert_mov_pack_b16x4<
    V: v2::reg::MovPackVariant<Args = (R<u16>, R<u16>, R<u16>, R<u16>), Output = R<u64>>,
>() {
}
fn assert_mov_pack_b32x4<
    V: v2::reg::MovPackVariant<Args = (R<u32>, R<u32>, R<u32>, R<u32>), Output = R<[u64; 2]>>,
>() {
}
fn assert_mov_unpack_b16x4<
    V: v2::reg::MovUnpackVariant<Args = R<u64>, Output = (R<u16>, R<u16>, R<u16>, R<u16>)>,
>() {
}
fn assert_mov_unpack_b32x4<
    V: v2::reg::MovUnpackVariant<Args = R<[u64; 2]>, Output = (R<u32>, R<u32>, R<u32>, R<u32>)>,
>() {
}
fn assert_mov_unpack_b64x2<
    V: v2::reg::MovUnpackVariant<Args = R<[u64; 2]>, Output = (R<u64>, R<u64>)>,
>() {
}
fn assert_cvt_pack_without_c<
    V: v2::reg::CvtPackVariant<Args = (R<i32>, R<i32>), Output = R<u32>>,
>() {
}
fn assert_cvt_pack_with_c<
    V: v2::reg::CvtPackVariant<Args = (R<i32>, R<i32>, R<u32>), Output = R<u32>>,
>() {
}
fn assert_vote_any<V: v2::warp::VoteSyncVariant<Args = (R<u32>, R<bool>), Output = R<bool>>>() {}

/// `tile::copy` is the one typed-tile entry point with no synchronous ABI
/// call, so it needs its own awaited pin: this fixes the exact positional
/// spelling (`warp, context, site, &destination, &source`) that generated
/// artifacts emit.
#[allow(dead_code)]
async fn call_public_typed_tile_copy(
    warp: &mut TestWarpEngine,
    context: ExecCtx,
    global: &MappedView<Global>,
    shared: &MappedView<Shared>,
) -> Result<(), EngineError> {
    v2::tile::copy::<TileCopy>(warp, context, SiteId::new(20), shared, global).await
}

#[allow(dead_code)]
fn call_public_remote_async_store(
    warp: &mut TestWarpEngine,
    context: ExecCtx,
    destination: Address<Shared>,
    barrier: Address<Shared>,
) -> Result<(), EngineError> {
    v2::async_copy::st_async::<RemoteAsyncStore>(
        warp,
        context,
        SiteId::new(21),
        (
            destination,
            barrier,
            R::splat(0_i64),
            std::array::from_fn(|index| R::splat(index as u32)),
        ),
    )
}

#[allow(dead_code)]
fn call_public_typed_copy_functions(
    warp: &mut TestWarpEngine,
    context: ExecCtx,
    global: &MappedView<Global>,
    shared: &MappedView<Shared>,
    register: &MappedView<Register>,
    tmem: &MappedView<Tmem>,
) -> Result<(), EngineError> {
    v2::tile::cp_async::<Classic>(warp, context, SiteId::new(10), shared, global, ())?;
    v2::tile::cp_async_bulk::<Bulk>(warp, context, SiteId::new(11), global, shared, ())?;
    v2::tile::cp_async_bulk_tensor::<Tensor>(warp, context, SiteId::new(12), global, shared, ())?;
    v2::tile::cp_reduce_async_bulk_tensor::<TensorReduce>(
        warp,
        context,
        SiteId::new(13),
        global,
        shared,
        (),
    )?;
    v2::tile::tcgen05_cp::<TileCp>(warp, context, SiteId::new(14), tmem, shared)?;
    v2::tile::tcgen05_ld::<TileLd>(warp, context, SiteId::new(15), register, tmem, ())?;
    v2::tile::tcgen05_st::<TileSt>(warp, context, SiteId::new(16), tmem, register, ())?;
    v2::tile::gemm_async::<DenseGemm>(
        warp,
        context,
        SiteId::new(17),
        tmem,
        shared,
        shared,
        R::splat(false),
    )
}

#[test]
fn public_typed_variants_are_nameable_from_an_external_frontend() {
    assert_copy::<TileCopy>();
    assert_classic::<Classic>();
    assert_bulk::<Bulk>();
    assert_tensor::<Tensor>();
    assert_reduce::<TensorReduce>();
    assert_cp::<TileCp>();
    assert_ld::<TileLd>();
    assert_st::<TileSt>();
    assert_gemm::<DenseGemm>();
    assert_gemm::<MixedBlockGemm>();
    assert_gemm::<DenseCta2BankedAGemm>();
    assert_gemm_ws::<DenseWsGemm>();
    assert_gemm_ws::<DenseWsBatchedGemm>();
    assert_raw_cp_async::<v2::async_copy::variant::CpAsync<16>>();
    assert_raw_cp_async::<v2::async_copy::variant::CpAsync<16, v2::async_copy::variant::SourceSize>>(
    );
    assert_raw_cp_async::<v2::async_copy::variant::CpAsync<16, v2::async_copy::variant::ZeroFill>>(
    );
    assert_raw_bulk_reduce::<
        v2::async_copy::variant::BulkS2gReduce<
            v2::reg::variant::F32,
            v2::async_copy::variant::ReduceAdd,
            v2::mem::variant::Sys,
        >,
    >();
    assert_remote_async_store::<RemoteAsyncStore>();
    assert_raw_bulk_reduce::<
        v2::async_copy::variant::BulkS2gReduce<
            v2::reg::variant::F64,
            v2::async_copy::variant::ReduceAdd,
            v2::mem::variant::Sys,
        >,
    >();

    fn consumes_external_map<M: ElementMap<Global, Runtime = usize>>(_: M) {}
    consumes_external_map(LinearMap);
}

#[test]
fn pinned_nymph_register_forms_are_nameable_from_an_external_frontend() {
    use v2::reg::variant as rv;

    assert_add_u16::<rv::F16Rn>();
    assert_add_u16::<rv::Bf16Rn>();
    assert_sub_u16::<rv::F16Rn>();
    assert_sub_u16::<rv::Bf16Rn>();
    assert_mul_u16::<rv::F16Rn>();
    assert_mul_u16::<rv::Bf16Rn>();
    assert_mul_wide_i16::<rv::MulWide<rv::I16>>();
    assert_mad_wide_u16::<rv::MadWide<rv::U16>>();
    assert_mad_f32::<rv::F32Arithmetic>();
    assert_mad_f64::<rv::F64Rn>();
    assert_mad_f64::<rv::F64Arithmetic<rv::Rz>>();
    assert_mad_f64::<rv::F64Arithmetic<rv::Rm>>();
    assert_mad_f64::<rv::F64Arithmetic<rv::Rp>>();
    assert_fma_u16::<rv::F16Rn>();
    assert_fma_u16::<rv::Bf16Rn>();
    assert_clmad_u64::<rv::ClmadLo>();
    assert_clmad_u64::<rv::ClmadHi>();
    assert_lop3::<rv::Lop3<0x1a>>();
    assert_lop3_bool::<rv::Lop3Bool<0x80, rv::BoolAnd>>();
    assert_lop3_bool::<rv::Lop3Bool<0xfe, rv::BoolOr>>();
    assert_div_i32::<rv::I32>();
    assert_div_u32::<rv::U32>();
    assert_div_i64::<rv::I64>();
    assert_div_u64::<rv::U64>();
    assert_min_u16::<rv::F16>();
    assert_min_u16::<rv::Bf16>();
    assert_min_u32::<rv::Bf16x2>();
    assert_max_u32::<rv::Bf16x2>();
    assert_min_u32::<rv::F16x2>();
    assert_max_u32::<rv::F16x2>();
    assert_and_bool::<rv::Pred>();
    assert_mov_pack_b16x2::<rv::B32>();
    assert_mov_pack_b32x2::<rv::B64>();
    assert_mov_pack_b64x2::<rv::B128>();
    assert_mov_unpack_b16x2::<rv::B32>();
    assert_mov_unpack_b32x2::<rv::B64>();
    assert_mov_pack_b16x4::<rv::B16x4>();
    assert_mov_pack_b32x4::<rv::B32x4>();
    assert_mov_unpack_b16x4::<rv::B16x4>();
    assert_mov_unpack_b32x4::<rv::B32x4>();
    assert_mov_unpack_b64x2::<rv::B64x2>();
    assert_cvt_pack_without_c::<rv::CvtPack<16, false>>();
    assert_cvt_pack_without_c::<rv::CvtPack<16, true>>();
    assert_cvt_pack_with_c::<rv::CvtPack<8, false>>();
    assert_cvt_pack_with_c::<rv::CvtPack<8, true>>();
    assert_cvt_pack_with_c::<rv::CvtPack<4, false>>();
    assert_cvt_pack_with_c::<rv::CvtPack<4, true>>();
    assert_cvt_pack_with_c::<rv::CvtPack<2, false>>();
    assert_cvt_pack_with_c::<rv::CvtPack<2, true>>();
    assert_max_f32_three::<rv::F32ThreeSource>();
    assert_max_u16::<rv::F16>();
    assert_max_u16::<rv::Bf16>();
    assert_cvt_down::<rv::Cvt<rv::F32, rv::F16, rv::Rn>>();
    assert_cvt_down::<rv::Cvt<rv::F32, rv::Bf16, rv::Rn>>();
    assert_cvt_up::<rv::Cvt<rv::F16, rv::F32>>();
    assert_cvt_up::<rv::Cvt<rv::Bf16, rv::F32>>();
    assert_cvt_f32::<rv::Cvt<rv::F32, rv::F32, rv::Rmi>>();
    assert_cvt_i32_f32::<rv::Cvt<rv::I32, rv::F32, rv::Rn>>();
    assert_cvt_u32_f32::<rv::Cvt<rv::U32, rv::F32, rv::Rn>>();
    assert_cvt_i64_f32::<rv::Cvt<rv::I64, rv::F32, rv::Rn>>();
    assert_cvt_u64_f32::<rv::Cvt<rv::U64, rv::F32, rv::Rn>>();

    // Packed FP8 cvt: one specialization per modifier combination the ISA
    // spells, named exactly as the generated artifact names it.
    assert_cvt_f32_pair_to_packed_byte::<
        rv::Cvt<rv::F32, rv::E4m3x2, rv::PackedMode<rv::Rn, rv::SatFinite, rv::NoRelu>>,
    >();
    assert_cvt_f32_pair_to_packed_byte::<
        rv::Cvt<rv::F32, rv::E5m2x2, rv::PackedMode<rv::Rn, rv::SatFinite, rv::Relu>>,
    >();
    assert_cvt_packed_word_to_packed_byte::<
        rv::Cvt<rv::F16x2, rv::E4m3x2, rv::PackedMode<rv::Rn, rv::SatFinite, rv::Relu>>,
    >();
    assert_cvt_packed_word_to_packed_byte::<
        rv::Cvt<rv::Bf16x2, rv::E5m2x2, rv::PackedMode<rv::Rn, rv::SatFinite, rv::NoRelu>>,
    >();
    assert_cvt_packed_byte_to_packed_word::<
        rv::Cvt<rv::E4m3x2, rv::F16x2, rv::PackedMode<rv::Rn, rv::NoSatFinite, rv::NoRelu>>,
    >();
    assert_cvt_packed_byte_to_packed_word::<
        rv::Cvt<rv::E5m2x2, rv::F16x2, rv::PackedMode<rv::Rn, rv::NoSatFinite, rv::Relu>>,
    >();
    assert_cvt_packed_byte_to_packed_word::<
        rv::Cvt<rv::E4m3x2, rv::Bf16x2, rv::PackedMode<rv::Rn, rv::NoSatFinite, rv::NoRelu>>,
    >();
    assert_cvt_packed_byte_to_packed_word::<
        rv::Cvt<rv::E5m2x2, rv::Bf16x2, rv::PackedMode<rv::Rn, rv::SatFinite, rv::Relu>>,
    >();

    // `.e2m1x2` uses the same byte carrier in both conversion directions.
    assert_cvt_f32_pair_to_nibble_pair::<
        rv::Cvt<rv::F32, rv::E2m1x2, rv::PackedMode<rv::Rn, rv::SatFinite, rv::NoRelu>>,
    >();
    assert_cvt_packed_word_to_nibble_pair::<
        rv::Cvt<rv::Bf16x2, rv::E2m1x2, rv::PackedMode<rv::Rn, rv::SatFinite, rv::Relu>>,
    >();
    assert_cvt_to_packed_word::<
        R<u8>,
        rv::Cvt<rv::E2m1x2, rv::F16x2, rv::PackedMode<rv::Rn, rv::NoSatFinite, rv::Relu>>,
    >();
    assert_cvt_to_packed_word::<
        R<u8>,
        rv::Cvt<rv::E2m1x2, rv::Bf16x2, rv::PackedMode<rv::Rn, rv::SatFinite, rv::NoRelu>>,
    >();

    assert_cvt_f32_pair_to_packed_byte::<
        rv::Cvt<rv::F32, rv::E2m3x2, rv::PackedMode<rv::Rz, rv::SatFinite, rv::NoRelu>>,
    >();
    assert_cvt_f32_pair_to_packed_byte::<
        rv::Cvt<
            rv::F32,
            rv::E3m2x2,
            rv::PackedMode<rv::Rn, rv::SatFinite, rv::Relu, rv::NoScale, rv::Pzo>,
        >,
    >();
    assert_cvt_scaled_packed_word_to_packed_byte::<
        rv::Cvt<
            rv::Bf16x2,
            rv::E2m3x2,
            rv::PackedMode<rv::Rz, rv::SatFinite, rv::NoRelu, rv::ScaledUe8m0N1, rv::Pzo>,
        >,
    >();
    assert_cvt_packed_byte_to_packed_word::<
        rv::Cvt<rv::Ue5m3x2, rv::F16x2, rv::PackedMode<rv::Rn, rv::NoSatFinite, rv::NoRelu>>,
    >();

    // `.rs` adds the `rbits` operand and groups four primaries.
    assert_cvt_f32_quad_to_packed_word::<
        rv::Cvt<rv::F32, rv::E4m3x4, rv::PackedMode<rv::Rs, rv::SatFinite, rv::NoRelu>>,
    >();
    assert_cvt_f32_quad_to_packed_word::<
        rv::Cvt<rv::F32, rv::E5m2x4, rv::PackedMode<rv::Rs, rv::SatFinite, rv::Relu>>,
    >();
    assert_cvt_f32_quad_to_nibble_quad::<
        rv::Cvt<rv::F32, rv::E2m1x4, rv::PackedMode<rv::Rs, rv::SatFinite, rv::NoRelu>>,
    >();

    // `.ue8m0x2` carries the rounding on the mode's first parameter.
    assert_cvt_f32_pair_to_packed_byte::<
        rv::Cvt<rv::F32, rv::Ue8m0x2, rv::PackedMode<rv::Rz, rv::NoSatFinite, rv::NoRelu>>,
    >();
    assert_cvt_packed_word_to_packed_byte::<
        rv::Cvt<rv::Bf16x2, rv::Ue8m0x2, rv::PackedMode<rv::Rp, rv::SatFinite, rv::NoRelu>>,
    >();
    assert_cvt_packed_byte_to_packed_word::<
        rv::Cvt<rv::Ue8m0x2, rv::Bf16x2, rv::PackedMode<rv::Rn, rv::NoSatFinite, rv::NoRelu>>,
    >();

    // `.scaled::n2::ue8m0` is the mode's fourth parameter and adds the scale
    // operand; every unscaled spelling above keeps naming only three.
    assert_cvt_scaled_to_packed_word::<
        rv::Cvt<
            rv::E4m3x2,
            rv::Bf16x2,
            rv::PackedMode<rv::Rn, rv::NoSatFinite, rv::NoRelu, rv::ScaledUe8m0N2>,
        >,
    >();
    assert_cvt_scaled_to_packed_word::<
        rv::Cvt<
            rv::E5m2x2,
            rv::Bf16x2,
            rv::PackedMode<rv::Rn, rv::SatFinite, rv::Relu, rv::ScaledUe8m0N2>,
        >,
    >();
    assert_cvt_to_packed_word::<
        (R<u8>, R<u16>),
        rv::Cvt<
            rv::E2m1x2,
            rv::Bf16x2,
            rv::PackedMode<rv::Rn, rv::SatFinite, rv::NoRelu, rv::ScaledUe8m0N2>,
        >,
    >();

    assert_setp_f32::<rv::Setp<rv::F32, rv::Lt>>();
    assert_setp_u16::<rv::Setp<rv::F16, rv::Lt>>();
    assert_setp_u16::<rv::Setp<rv::Bf16, rv::Lt>>();
    assert_set_u32::<rv::Set<rv::F32, rv::Neu, rv::U32, rv::Ftz>>();
    assert_set_packed::<rv::SetPacked<rv::U8, rv::Eq>>();
    assert_set_bool_u32::<rv::SetBool<rv::F16x2, rv::Nan, rv::U32, rv::BoolXor, rv::Ftz>>();
    assert_setp_bool::<rv::SetpBool<rv::Bf16, rv::Eq, rv::BoolAnd>>();
    assert_setp_pair::<rv::SetpPair<rv::F16x2, rv::Ltu, rv::Ftz>>();
    assert_setp_pair_bool::<rv::SetpPairBool<rv::F32, rv::Ge, rv::BoolOr, rv::Ftz>>();
    assert_slct_u32::<rv::Slct<rv::B32, rv::F32, rv::Ftz>>();
    assert_testp_f32::<rv::Testp<rv::F32, rv::Subnormal>>();
}

#[test]
fn pinned_nymph_scalar_and_scope_forms_are_nameable_from_an_external_frontend() {
    use v2::reg::variant as rv;

    assert_add_i32::<rv::I32>();
    assert_sub_i32::<rv::I32>();
    assert_mul_i32::<rv::I32>();
    assert_min_i32::<rv::I32>();
    assert_max_i32::<rv::I32>();
    assert_and_i32::<rv::I32>();
    assert_or_i32::<rv::I32>();
    assert_xor_i32::<rv::I32>();
    assert_shl_i32::<rv::I32>();
    // `shr.b32`/`shr.b64` carry the unsigned scalar because PTX untyped shifts
    // are logical; the bit-size markers must stay reachable from outside.
    assert_shr_u32::<rv::B32>();
    assert_shr_u64::<rv::B64>();
    assert_shr_u32::<rv::U32>();
    assert_shr_u64::<rv::U64>();
    assert_neg_i32::<rv::I32>();
    assert_selp_i32::<rv::I32>();
    assert_selp_u16::<rv::B16>();

    assert_setp_f32::<rv::Setp<rv::F32, rv::Eq>>();
    assert_setp_f32::<rv::Setp<rv::F32, rv::Ne>>();
    assert_setp_f32::<rv::Setp<rv::F32, rv::Lt>>();
    assert_setp_f32::<rv::Setp<rv::F32, rv::Le>>();
    assert_setp_f32::<rv::Setp<rv::F32, rv::Gt>>();
    assert_setp_f32::<rv::Setp<rv::F32, rv::Ge>>();

    assert_mov_u32::<rv::LaneId>();
    assert_mov_u32::<rv::WarpIdInCta>();
    assert_mov_u32::<rv::CtaId>();
    assert_mov_u32::<rv::ClusterId>();
    assert_vote_any::<v2::warp::variant::Any>();
}
