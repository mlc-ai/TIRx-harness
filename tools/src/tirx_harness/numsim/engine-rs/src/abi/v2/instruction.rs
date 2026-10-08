//! Macros for sealed compile-time instruction specializations.

// One declaration owns a variant's types, bounds and execution implementation.
// The method stays explicit: register, synchronous and awaited instructions
// keep their different signatures and engine-access contracts.
macro_rules! instruction_variant {
    ([$($implementation:tt)*] $spec:ident, $variant:ty $(where [$($bounds:tt)*])?,
     $args:ty => $output:ty;
     $execute:item) => {
        $($implementation)* $spec::sealed::Sealed for $variant $(where $($bounds)*)? {}
        $($implementation)* $spec::Variant for $variant $(where $($bounds)*)? {
            type Args = $args;
            type Output = $output;
        }
        $($implementation)* $spec::sealed::Execute for $variant $(where $($bounds)*)? {
            $execute
        }
    };
}

macro_rules! sync_instruction {
    ($spec:ident, $variant:ident, $function:ident) => {
        mod $spec {
            pub(super) mod sealed {
                pub trait Sealed {}

                pub trait Execute: super::Variant {
                    fn execute(
                        engine: &mut $crate::abi::v2::Engine,
                        context: $crate::abi::v2::ExecCtx,
                        site: $crate::abi::v2::SiteId,
                        args: Self::Args,
                    ) -> Result<Self::Output, $crate::abi::v2::EngineError>;
                }
            }

            #[allow(private_bounds)]
            pub trait Variant: sealed::Sealed {
                type Args;
                type Output;
            }
        }

        pub use $spec::Variant as $variant;

        #[allow(private_bounds)]
        // Keep the closed specialization as a codegen boundary. Large generated
        // kernels can contain thousands of calls to the same variant; forcing
        // this adapter into every call site makes LLVM repeatedly optimize the
        // same instruction glue without exposing useful cross-call constants.
        #[inline(never)]
        pub fn $function<V>(
            engine: &mut $crate::abi::v2::Engine,
            context: $crate::abi::v2::ExecCtx,
            site: $crate::abi::v2::SiteId,
            args: V::Args,
        ) -> Result<V::Output, $crate::abi::v2::EngineError>
        where
            V: $variant + $spec::sealed::Execute,
        {
            #[cfg(feature = "profile")]
            $crate::instruction_profile::record(context, site, std::any::type_name::<V>());
            <V as $spec::sealed::Execute>::execute(engine, context, site, args)
        }
    };
}

/// Define a register-only instruction.
///
/// The specialization implementation deliberately has no generic warp
/// parameter or engine handle. Register semantics depend only on their
/// operands plus the execution context/site when required, so the engine
/// crate can compile the concrete implementation once instead of every
/// generated artifact monomorphizing it again.
macro_rules! register_instruction {
    ($spec:ident, $variant:ident, $function:ident) => {
        mod $spec {
            pub(super) mod sealed {
                pub trait Sealed {}

                pub trait Execute: super::Variant {
                    fn execute(
                        context: $crate::abi::v2::ExecCtx,
                        site: $crate::abi::v2::SiteId,
                        args: Self::Args,
                    ) -> Result<Self::Output, $crate::abi::v2::EngineError>;
                }
            }

            #[allow(private_bounds)]
            pub trait Variant: sealed::Sealed {
                type Args;
                type Output;
            }
        }

        pub use $spec::Variant as $variant;

        #[allow(private_bounds)]
        // See `sync_instruction`: one monomorphized adapter per closed variant
        // is substantially cheaper to optimize than duplicating it at every
        // generated instruction site.
        #[inline(never)]
        pub fn $function<V>(
            context: $crate::abi::v2::ExecCtx,
            site: $crate::abi::v2::SiteId,
            args: V::Args,
        ) -> Result<V::Output, $crate::abi::v2::EngineError>
        where
            V: $variant + $spec::sealed::Execute,
        {
            #[cfg(feature = "profile")]
            $crate::instruction_profile::record(context, site, std::any::type_name::<V>());
            <V as $spec::sealed::Execute>::execute(context, site, args)
        }
    };
}

/// Define an instruction whose runtime operand carrier can be borrowed or
/// owned while the compile-time specialization and result type stay fixed.
/// This is used by `ld`/`st`: both a raw `Address` and a borrowed bound-buffer
/// address denote the same PTX instruction rather than separate ABI calls.
macro_rules! sync_instruction_generic_args {
    ($spec:ident, $variant:ident, $function:ident) => {
        mod $spec {
            pub(super) mod sealed {
                pub trait Sealed {}

                pub trait Execute<W, Args>: super::Variant
                where
                    W: $crate::abi::v2::WarpHandle,
                {
                    fn execute(
                        warp: &mut W,
                        context: $crate::abi::v2::ExecCtx,
                        site: $crate::abi::v2::SiteId,
                        args: Args,
                    ) -> Result<Self::Output, $crate::abi::v2::EngineError>;
                }

                pub trait Argument<V: super::Variant, W: $crate::abi::v2::WarpHandle> {
                    fn execute(
                        self,
                        warp: &mut W,
                        context: $crate::abi::v2::ExecCtx,
                        site: $crate::abi::v2::SiteId,
                    ) -> Result<V::Output, $crate::abi::v2::EngineError>;
                }

                impl<V, W, Args> Argument<V, W> for Args
                where
                    V: super::Variant + Execute<W, Args>,
                    W: $crate::abi::v2::WarpHandle,
                {
                    #[inline(always)]
                    fn execute(
                        self,
                        warp: &mut W,
                        context: $crate::abi::v2::ExecCtx,
                        site: $crate::abi::v2::SiteId,
                    ) -> Result<V::Output, $crate::abi::v2::EngineError> {
                        <V as Execute<W, Args>>::execute(warp, context, site, self)
                    }
                }
            }

            #[allow(private_bounds)]
            pub trait Variant: sealed::Sealed {
                type Output;
            }
        }

        pub use $spec::Variant as $variant;

        #[allow(private_bounds)]
        // Preserve the load/store specialization boundary instead of expanding
        // the generic argument adapter at every generated memory access.
        #[inline(never)]
        pub fn $function<V>(
            warp: &mut $crate::abi::v2::Engine,
            context: $crate::abi::v2::ExecCtx,
            site: $crate::abi::v2::SiteId,
            args: impl $spec::sealed::Argument<V, $crate::abi::v2::Engine>,
        ) -> Result<V::Output, $crate::abi::v2::EngineError>
        where
            V: $variant,
        {
            #[cfg(feature = "profile")]
            $crate::instruction_profile::record(context, site, std::any::type_name::<V>());
            $spec::sealed::Argument::<V, $crate::abi::v2::Engine>::execute(
                args, warp, context, site,
            )
        }
    };
}

/// Define an awaited instruction.
///
/// The default form carries one runtime operand carrier (`Self::Args`). The
/// `no_args` form declares an instruction whose entire operand set is the
/// compile-time specialization, so the ABI call takes only the
/// `(engine, context, site)` prelude. That is not a convenience spelling: an
/// operand-less mnemonic has no carrier to name, and forcing it to accept `()`
/// would make the generated call site state an operand the instruction does
/// not have.
macro_rules! async_instruction {
    ($spec:ident, $variant:ident, $function:ident) => {
        $crate::abi::v2::instruction::async_instruction!(
            @define $spec, $variant, $function, [args: Args]
        );
    };
    ($spec:ident, $variant:ident, $function:ident, no_args) => {
        $crate::abi::v2::instruction::async_instruction!(
            @define $spec, $variant, $function, []
        );
    };
    (@define $spec:ident, $variant:ident, $function:ident, [$($operand:ident: $carrier:ident)?]) => {
        mod $spec {
            pub(super) mod sealed {
                pub trait Sealed {}

                pub trait Execute: super::Variant {
                    fn execute(
                        engine: &mut $crate::abi::v2::Engine,
                        context: $crate::abi::v2::ExecCtx,
                        site: $crate::abi::v2::SiteId,
                        $($operand: Self::$carrier,)?
                    ) -> impl std::future::Future<
                        Output = Result<Self::Output, $crate::abi::v2::EngineError>,
                    > + Send;
                }
            }

            #[allow(private_bounds)]
            pub trait Variant: sealed::Sealed {
                $(type $carrier;)?
                type Output;
            }
        }

        pub use $spec::Variant as $variant;

        #[allow(private_bounds)]
        #[inline(never)]
        pub async fn $function<V>(
            engine: &mut $crate::abi::v2::Engine,
            context: $crate::abi::v2::ExecCtx,
            site: $crate::abi::v2::SiteId,
            $($operand: V::$carrier,)?
        ) -> Result<V::Output, $crate::abi::v2::EngineError>
        where
            V: $variant + $spec::sealed::Execute,
        {
            #[cfg(feature = "profile")]
            $crate::instruction_profile::record(context, site, std::any::type_name::<V>());
            <V as $spec::sealed::Execute>::execute(engine, context, site, $($operand,)?).await
        }
    };
}

/// Define a whole-tile instruction whose operands are pure mapped views.
///
/// See [`async_mapped_instruction`] for why the views are declared as
/// associated spaces rather than folded into the runtime carrier. This is the
/// synchronous form: it optionally carries one runtime operand after the view
/// list, spelled `args` at the declaration site.
macro_rules! mapped_instruction {
    (
        $spec:ident, $variant:ident, $function:ident,
        views: [$($view:ident: $space:ident),+ $(,)?]
    ) => {
        $crate::abi::v2::instruction::mapped_instruction!(
            @define $spec, $variant, $function, [$($view: $space),+], []
        );
    };
    (
        $spec:ident, $variant:ident, $function:ident,
        views: [$($view:ident: $space:ident),+ $(,)?], args
    ) => {
        $crate::abi::v2::instruction::mapped_instruction!(
            @define $spec, $variant, $function, [$($view: $space),+], [args: Args]
        );
    };
    (
        @define $spec:ident, $variant:ident, $function:ident,
        [$($view:ident: $space:ident),+], [$($operand:ident: $carrier:ident)?]
    ) => {
        mod $spec {
            pub(super) mod sealed {
                pub trait Sealed {}

                pub trait Execute<W>: super::Variant
                where
                    W: $crate::abi::v2::WarpHandle,
                {
                    fn execute(
                        warp: &mut W,
                        context: $crate::abi::v2::ExecCtx,
                        site: $crate::abi::v2::SiteId,
                        $($view: &$crate::abi::v2::MappedView<Self::$space>,)+
                        $($operand: Self::$carrier,)?
                    ) -> Result<Self::Output, $crate::abi::v2::EngineError>;
                }
            }

            #[allow(private_bounds)]
            pub trait Variant: sealed::Sealed {
                $(
                    #[doc(hidden)]
                    type $space: $crate::abi::v2::MemorySpace;
                )+
                $(type $carrier;)?
                type Output;
            }
        }

        pub use $spec::Variant as $variant;

        #[allow(private_bounds)]
        #[inline(always)]
        pub fn $function<V>(
            warp: &mut $crate::abi::v2::Engine,
            context: $crate::abi::v2::ExecCtx,
            site: $crate::abi::v2::SiteId,
            $($view: &$crate::abi::v2::MappedView<V::$space>,)+
            $($operand: V::$carrier,)?
        ) -> Result<V::Output, $crate::abi::v2::EngineError>
        where
            V: $variant + $spec::sealed::Execute<$crate::abi::v2::Engine>,
        {
            #[cfg(feature = "profile")]
            $crate::instruction_profile::record(context, site, std::any::type_name::<V>());
            <V as $spec::sealed::Execute<$crate::abi::v2::Engine>>::execute(
                warp,
                context,
                site,
                $($view,)+
                $($operand,)?
            )
        }
    };
}

/// Define an awaited whole-tile instruction whose operands are pure mapped
/// views.
///
/// A tile operation's views *are* its operand list: the frontend owns the
/// element maps, and the state space each view addresses is a compile-time
/// property of the specialization. The spaces are therefore declared as
/// associated types on the `Variant` trait and the views appear positionally
/// in the ABI call, exactly as a PTX operand list would — not folded into the
/// runtime carrier, which would let a generated call site name a view whose
/// space the specialization does not fix.
///
/// The execution hook stays generic over the warp handle. Tile bodies are
/// instantiated once per engine mode by `for_each_engine_mode!`, while the ABI
/// entry point binds the single concrete [`crate::abi::v2::Engine`] of the
/// build profile; keeping `Execute` mode-generic leaves that lattice exactly
/// where it is instead of folding two axes in one step.
macro_rules! async_mapped_instruction {
    (
        $spec:ident, $variant:ident, $function:ident,
        views: [$($view:ident: $space:ident),+ $(,)?]
    ) => {
        $crate::abi::v2::instruction::async_mapped_instruction!(
            @define $spec, $variant, $function, [$($view: $space),+], []
        );
    };
    (
        @define $spec:ident, $variant:ident, $function:ident,
        [$($view:ident: $space:ident),+], [$($operand:ident: $carrier:ident)?]
    ) => {
        mod $spec {
            pub(super) mod sealed {
                pub trait Sealed {}

                pub trait Execute<W>: super::Variant
                where
                    W: $crate::abi::v2::WarpHandle + Send,
                {
                    #[allow(async_fn_in_trait)]
                    async fn execute(
                        warp: &mut W,
                        context: $crate::abi::v2::ExecCtx,
                        site: $crate::abi::v2::SiteId,
                        $($view: &$crate::abi::v2::MappedView<Self::$space>,)+
                        $($operand: Self::$carrier,)?
                    ) -> Result<Self::Output, $crate::abi::v2::EngineError>;
                }
            }

            #[allow(private_bounds)]
            pub trait Variant: sealed::Sealed {
                $(
                    #[doc(hidden)]
                    type $space: $crate::abi::v2::MemorySpace;
                )+
                $(type $carrier;)?
                type Output;
            }
        }

        pub use $spec::Variant as $variant;

        #[allow(private_bounds)]
        #[inline(never)]
        pub async fn $function<V>(
            warp: &mut $crate::abi::v2::Engine,
            context: $crate::abi::v2::ExecCtx,
            site: $crate::abi::v2::SiteId,
            $($view: &$crate::abi::v2::MappedView<V::$space>,)+
            $($operand: V::$carrier,)?
        ) -> Result<V::Output, $crate::abi::v2::EngineError>
        where
            V: $variant + $spec::sealed::Execute<$crate::abi::v2::Engine>,
        {
            #[cfg(feature = "profile")]
            $crate::instruction_profile::record(context, site, std::any::type_name::<V>());
            <V as $spec::sealed::Execute<$crate::abi::v2::Engine>>::execute(
                warp,
                context,
                site,
                $($view,)+
                $($operand,)?
            )
            .await
        }
    };
}

pub(crate) use async_instruction;
pub(crate) use async_mapped_instruction;
pub(crate) use instruction_variant;
pub(crate) use mapped_instruction;
pub(crate) use register_instruction;
pub(crate) use sync_instruction;
pub(crate) use sync_instruction_generic_args;
