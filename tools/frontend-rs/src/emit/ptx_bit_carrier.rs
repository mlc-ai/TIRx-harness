//! Exact PTX bit-size register carriers.

use tvm::tvm_ffi::object::ObjectRef;

use super::super::analyze::util::{dtype_of, ffi_error, unsupported, AResult, Failure};
use super::{Emitter, RustValue, Uniformity};
use crate::emit::register_call::bit_carrier_dtypes;
use crate::tables::dtype_itemsize;

fn unsigned_rust_type(width: i64) -> String {
    if width == 128 {
        "U64x2".to_owned()
    } else {
        format!("u{width}")
    }
}

impl<'a> Emitter<'a> {
    /// `emit_as_unsigned_bits`: materialize a legal `.bN` carrier as its
    /// unsigned storage payload.
    pub fn emit_as_unsigned_bits(
        &mut self,
        value: &ObjectRef,
        width: i64,
        op_name: &str,
        prefix: &str,
        access_mask: Option<&str>,
    ) -> AResult<RustValue> {
        let unsigned_type = unsigned_rust_type(width);
        let logical_dtype = dtype_of(value)?;
        if !bit_carrier_dtypes(width).contains(&logical_dtype.as_str()) {
            return unsupported(format!(
                "{op_name} expects a b{width} carrier, got {logical_dtype}"
            ));
        }
        let emitted = match access_mask {
            None => {
                let emitted = self.emit_expr(value)?;
                self.observe_pointer_bits(emitted)
            }
            Some(mask) => {
                let previous = self.register_access_mask.replace(mask.to_owned());
                let result = self
                    .emit_expr(value)
                    .map(|emitted| self.observe_pointer_bits(emitted));
                self.register_access_mask = previous;
                result?
            }
        };
        let emitted = self.as_warp_value(emitted);
        if emitted.rust_type == unsigned_type {
            return Ok(emitted);
        }
        let conversion = match (logical_dtype.as_str(), emitted.rust_type.as_str()) {
            ("int8", "i8") => format!("{}[lane] as u8", emitted.code),
            ("int16", "i16") => format!("{}[lane] as u16", emitted.code),
            ("float16", "f32") => format!("decoded_fp16_to_bits({}[lane])", emitted.code),
            ("bfloat16", "f32") => format!("decoded_bf16_to_bits({}[lane])", emitted.code),
            ("int32", "i32") => format!("{}[lane] as u32", emitted.code),
            ("float32", "f32") => format!("{}[lane].to_bits()", emitted.code),
            ("int64", "i64") => format!("{}[lane] as u64", emitted.code),
            ("float64", "f64") => format!("{}[lane].to_bits()", emitted.code),
            _ => {
                return unsupported(format!(
                    "{op_name} {logical_dtype} carrier lowered to {}, expected an exact b{width} representation",
                    emitted.rust_type
                ))
            }
        };
        let bits = self.control_name(prefix);
        self.emit_line(&format!(
            "let {bits} = WarpValue::from_fn(|lane| {conversion});"
        ));
        Ok(RustValue::new(bits, unsigned_type, Uniformity::Varying))
    }

    /// `emit_from_unsigned_bits`: reinterpret unsigned storage bits as one
    /// exact PTX carrier.
    pub fn emit_from_unsigned_bits(
        &mut self,
        bits: RustValue,
        destination_dtype: &str,
        width: i64,
        op_name: &str,
        prefix: &str,
    ) -> AResult<RustValue> {
        let unsigned_type = unsigned_rust_type(width);
        if bits.rust_type != unsigned_type {
            return unsupported(format!(
                "{op_name} result lowered to {}, expected {unsigned_type} bits",
                bits.rust_type
            ));
        }
        if !bit_carrier_dtypes(width).contains(&destination_dtype) {
            return unsupported(format!(
                "{op_name} destination {destination_dtype} has no exact b{width} representation"
            ));
        }
        if destination_dtype == format!("uint{width}") || width == 128 {
            return Ok(bits);
        }
        let (conversion, rust_type) = match (width, destination_dtype) {
            (8, "int8") => (format!("{}[lane] as i8", bits.code), "i8"),
            (16, "int16") => (format!("{}[lane] as i16", bits.code), "i16"),
            (32, "int32") => (format!("{}[lane] as i32", bits.code), "i32"),
            (32, "float32") => (format!("f32::from_bits({}[lane])", bits.code), "f32"),
            (64, "int64") => (format!("{}[lane] as i64", bits.code), "i64"),
            (64, "float64") => (format!("f64::from_bits({}[lane])", bits.code), "f64"),
            _ => {
                return unsupported(format!(
                    "{op_name} destination {destination_dtype} has no exact b{width} representation"
                ))
            }
        };
        let value = self.control_name(prefix);
        self.emit_line(&format!(
            "let {value} = WarpValue::from_fn(|lane| {conversion});"
        ));
        Ok(RustValue::new(value, rust_type, Uniformity::Varying))
    }

    /// `emit_extended_register_value`: extend instruction result bits, then
    /// reinterpret the register carrier.
    ///
    /// PTX relaxed destination typing is shared by loads and scalar CVT. The
    /// instruction's signedness, not the carrier's spelling, controls extension.
    /// The optional storage override keeps half/BF16 payloads out of f32 casts.
    pub fn emit_extended_register_value(
        &mut self,
        value: RustValue,
        dtype: &str,
        width: i64,
        signed: bool,
        op_name: &str,
        prefix: &str,
    ) -> AResult<(RustValue, Option<&'static str>)> {
        let Some(itemsize) = dtype_itemsize(self.ctx.schema, dtype) else {
            return Err(Failure::Ffi(ffi_error(&format!(
                "{op_name} destination {dtype} has no itemsize"
            ))));
        };
        let destination_width = itemsize * 8;
        if destination_width < width {
            return unsupported(format!("{op_name} destination is narrower than its result"));
        }
        let result_type = unsigned_rust_type(destination_width);
        let mut value = value;
        if width != 128 && (signed || value.rust_type != result_type) {
            let mut atom = format!("{}[lane]", value.code);
            if value.rust_type == "f32" || value.rust_type == "f64" {
                atom.push_str(".to_bits()");
            }
            if signed {
                atom = format!("({atom} as i{width})");
            }
            let conversion = if destination_width == 128 {
                if signed {
                    format!("[{atom} as u64, (({atom} as i64) >> 63) as u64]")
                } else {
                    format!("[{atom} as u64, 0_u64]")
                }
            } else {
                format!("{atom} as {result_type}")
            };
            let name = self.control_name(prefix);
            self.emit_line(&format!(
                "let {name} = WarpValue::from_fn(|lane| {conversion});"
            ));
            value = RustValue::new(name, result_type, Uniformity::Varying);
        }
        if dtype == "float16" || dtype == "bfloat16" {
            return Ok((value, Some("uint16")));
        }
        Ok((
            self.emit_from_unsigned_bits(
                value,
                dtype,
                destination_width,
                op_name,
                &format!("{prefix}_bits"),
            )?,
            None,
        ))
    }
}
