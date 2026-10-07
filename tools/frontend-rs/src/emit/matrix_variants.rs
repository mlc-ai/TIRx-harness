//! Flat register-signature mappings for the matrix engine ABI.
//! Each entry identifies an engine specialization and its register carriers.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DenseMmaEngine {
    F32B16,
    F16F16,
    F32Tf32,
    F32F8,
    F16F8,
    F64,
    PackedInteger,
    M8n8k4F16,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SparseMmaEngine {
    B16,
    Tf32,
    PackedInteger,
    F8,
}

/// Register counts and specialization family of one dense engine entry.
pub struct DenseMmaVariant {
    pub d_count: i64,
    pub a_count: i64,
    pub b_count: i64,
    pub c_count: i64,
    pub engine: DenseMmaEngine,
}

/// Register counts and specialization family of one sparse engine entry.
pub struct SparseMmaVariant {
    pub a_count: i64,
    pub b_count: i64,
    pub c_count: i64,
    pub engine: SparseMmaEngine,
}

/// Semantic operand types, independent of their packed register carriers.
pub struct MmaTypes<'a> {
    pub d: &'a str,
    pub a: &'a str,
    pub b: &'a str,
    pub c: &'a str,
}

impl MmaTypes<'_> {
    fn dense_entry(
        &self,
        shape: (i64, i64, i64),
        layouts: (&str, &str),
        saturate: bool,
        bit_op: Option<&str>,
    ) -> Option<(DenseMmaVariant, bool)> {
        let ((d_count, a_count, b_count, c_count), engine, legacy) = match (
            shape,
            layouts,
            (self.d, self.a, self.b, self.c),
            (saturate, bit_op),
        ) {
            (
                (8, 8, 4),
                ("row" | "col", "row" | "col"),
                ("float16", "float16", "float16", "float16"),
                (false, None),
            ) => ((4, 2, 2, 4), DenseMmaEngine::M8n8k4F16, true),
            (
                (8, 8, 4),
                ("row" | "col", "row" | "col"),
                ("float32", "float16", "float16", "float16"),
                (false, None),
            ) => ((8, 2, 2, 4), DenseMmaEngine::M8n8k4F16, false),
            (
                (8, 8, 4),
                ("row" | "col", "row" | "col"),
                ("float32", "float16", "float16", "float32"),
                (false, None),
            ) => ((8, 2, 2, 8), DenseMmaEngine::M8n8k4F16, true),
            (
                (16, 8, 8),
                ("row", "col"),
                ("float32", "float16", "float16", "float32"),
                (false, None),
            ) => ((4, 2, 1, 4), DenseMmaEngine::F32B16, true),
            (
                (16, 8, 8),
                ("row", "col"),
                ("float32", "bfloat16", "bfloat16", "float32"),
                (false, None),
            ) => ((4, 2, 1, 4), DenseMmaEngine::F32B16, true),
            (
                (16, 8, 8),
                ("row", "col"),
                ("float16", "float16", "float16", "float16"),
                (false, None),
            ) => ((2, 2, 1, 2), DenseMmaEngine::F16F16, true),
            (
                (16, 8, 16),
                ("row", "col"),
                ("float32", "float16", "float16", "float32"),
                (false, None),
            ) => ((4, 4, 2, 4), DenseMmaEngine::F32B16, true),
            (
                (16, 8, 16),
                ("row", "col"),
                ("float32", "bfloat16", "bfloat16", "float32"),
                (false, None),
            ) => ((4, 4, 2, 4), DenseMmaEngine::F32B16, true),
            (
                (16, 8, 16),
                ("row", "col"),
                ("float16", "float16", "float16", "float16"),
                (false, None),
            ) => ((2, 4, 2, 2), DenseMmaEngine::F16F16, true),
            ((16, 8, 4), ("row", "col"), ("float32", "tf32", "tf32", "float32"), (false, None)) => {
                ((4, 2, 1, 4), DenseMmaEngine::F32Tf32, true)
            }
            ((16, 8, 8), ("row", "col"), ("float32", "tf32", "tf32", "float32"), (false, None)) => {
                ((4, 4, 2, 4), DenseMmaEngine::F32Tf32, true)
            }
            (
                (16, 8, 16),
                ("row", "col"),
                (
                    "float16",
                    "float8_e4m3fn" | "float8_e5m2",
                    "float8_e4m3fn" | "float8_e5m2",
                    "float16",
                ),
                (false, None),
            ) => ((2, 2, 1, 2), DenseMmaEngine::F16F8, false),
            (
                (16, 8, 16),
                ("row", "col"),
                (
                    "float32",
                    "float8_e4m3fn" | "float8_e5m2",
                    "float8_e4m3fn" | "float8_e5m2",
                    "float32",
                ),
                (false, None),
            ) => ((4, 2, 1, 4), DenseMmaEngine::F32F8, false),
            (
                (16, 8, 32),
                ("row", "col"),
                (
                    "float16",
                    "float8_e4m3fn" | "float8_e5m2",
                    "float8_e4m3fn" | "float8_e5m2",
                    "float16",
                ),
                (false, None),
            ) => ((2, 4, 2, 2), DenseMmaEngine::F16F8, false),
            (
                (16, 8, 32),
                ("row", "col"),
                (
                    "float32",
                    "float8_e4m3fn" | "float8_e5m2",
                    "float8_e4m3fn" | "float8_e5m2",
                    "float32",
                ),
                (false, None),
            ) => ((4, 4, 2, 4), DenseMmaEngine::F32F8, true),
            (
                (8, 8, 4),
                ("row", "col"),
                ("float64", "float64", "float64", "float64"),
                (false, None),
            ) => ((2, 1, 1, 2), DenseMmaEngine::F64, true),
            (
                (16, 8, 4),
                ("row", "col"),
                ("float64", "float64", "float64", "float64"),
                (false, None),
            ) => ((4, 2, 1, 4), DenseMmaEngine::F64, false),
            (
                (16, 8, 8),
                ("row", "col"),
                ("float64", "float64", "float64", "float64"),
                (false, None),
            ) => ((4, 4, 2, 4), DenseMmaEngine::F64, false),
            (
                (16, 8, 16),
                ("row", "col"),
                ("float64", "float64", "float64", "float64"),
                (false, None),
            ) => ((4, 8, 4, 4), DenseMmaEngine::F64, false),
            (
                (8, 8, 16),
                ("row", "col"),
                ("int32", "int8" | "uint8", "int8" | "uint8", "int32"),
                (_, None),
            ) => ((2, 1, 1, 2), DenseMmaEngine::PackedInteger, true),
            (
                (16, 8, 16),
                ("row", "col"),
                ("int32", "int8" | "uint8", "int8" | "uint8", "int32"),
                (_, None),
            ) => ((4, 2, 1, 4), DenseMmaEngine::PackedInteger, true),
            (
                (16, 8, 32),
                ("row", "col"),
                ("int32", "int8" | "uint8", "int8" | "uint8", "int32"),
                (_, None),
            ) => ((4, 4, 2, 4), DenseMmaEngine::PackedInteger, true),
            (
                (8, 8, 32),
                ("row", "col"),
                ("int32", "int4" | "uint4", "int4" | "uint4", "int32"),
                (_, None),
            ) => ((2, 1, 1, 2), DenseMmaEngine::PackedInteger, true),
            (
                (16, 8, 32),
                ("row", "col"),
                ("int32", "int4" | "uint4", "int4" | "uint4", "int32"),
                (_, None),
            ) => ((4, 2, 1, 4), DenseMmaEngine::PackedInteger, true),
            (
                (16, 8, 64),
                ("row", "col"),
                ("int32", "int4" | "uint4", "int4" | "uint4", "int32"),
                (_, None),
            ) => ((4, 4, 2, 4), DenseMmaEngine::PackedInteger, true),
            (
                (8, 8, 128),
                ("row", "col"),
                ("int32", "int1", "int1", "int32"),
                (false, Some("xor" | "and")),
            ) => ((2, 1, 1, 2), DenseMmaEngine::PackedInteger, true),
            (
                (16, 8, 128),
                ("row", "col"),
                ("int32", "int1", "int1", "int32"),
                (false, Some("xor" | "and")),
            ) => ((4, 2, 1, 4), DenseMmaEngine::PackedInteger, true),
            (
                (16, 8, 256),
                ("row", "col"),
                ("int32", "int1", "int1", "int32"),
                (false, Some("xor" | "and")),
            ) => ((4, 4, 2, 4), DenseMmaEngine::PackedInteger, true),
            _ => return None,
        };
        Some((
            DenseMmaVariant {
                d_count,
                a_count,
                b_count,
                c_count,
                engine,
            },
            legacy,
        ))
    }

    pub fn dense(
        &self,
        shape: (i64, i64, i64),
        layouts: (&str, &str),
        saturate: bool,
        bit_op: Option<&str>,
    ) -> Option<DenseMmaVariant> {
        self.dense_entry(shape, layouts, saturate, bit_op)
            .map(|(entry, _)| entry)
    }

    /// The legacy ABI exposes the entries marked by the lookup table.
    pub fn legacy_dense(
        &self,
        shape: (i64, i64, i64),
        layouts: (&str, &str),
        saturate: bool,
        bit_op: Option<&str>,
    ) -> Option<DenseMmaVariant> {
        self.dense_entry(shape, layouts, saturate, bit_op)
            .filter(|(_, legacy)| *legacy)
            .map(|(entry, _)| entry)
    }

    pub fn sparse(&self, k: i64, saturate: bool) -> Option<SparseMmaVariant> {
        let ((a_count, b_count, c_count), engine) =
            match (k, (self.d, self.a, self.b, self.c), saturate) {
                (16, ("float16", "float16", "float16", "float16"), false) => {
                    ((2, 2, 2), SparseMmaEngine::B16)
                }
                (16, ("float32", "float16", "float16", "float32"), false) => {
                    ((2, 2, 4), SparseMmaEngine::B16)
                }
                (16, ("float32", "bfloat16", "bfloat16", "float32"), false) => {
                    ((2, 2, 4), SparseMmaEngine::B16)
                }
                (32, ("float16", "float16", "float16", "float16"), false) => {
                    ((4, 4, 2), SparseMmaEngine::B16)
                }
                (32, ("float32", "float16", "float16", "float32"), false) => {
                    ((4, 4, 4), SparseMmaEngine::B16)
                }
                (32, ("float32", "bfloat16", "bfloat16", "float32"), false) => {
                    ((4, 4, 4), SparseMmaEngine::B16)
                }
                (8, ("float32", "tf32", "tf32", "float32"), false) => {
                    ((2, 2, 4), SparseMmaEngine::Tf32)
                }
                (16, ("float32", "tf32", "tf32", "float32"), false) => {
                    ((4, 4, 4), SparseMmaEngine::Tf32)
                }
                (
                    64,
                    (
                        "float32",
                        "float8_e4m3fn" | "float8_e5m2",
                        "float8_e4m3fn" | "float8_e5m2",
                        "float32",
                    ),
                    false,
                ) => ((4, 4, 4), SparseMmaEngine::F8),
                (32, ("int32", "int8" | "uint8", "int8" | "uint8", "int32"), _) => {
                    ((2, 2, 4), SparseMmaEngine::PackedInteger)
                }
                (64, ("int32", "int8" | "uint8", "int8" | "uint8", "int32"), _) => {
                    ((4, 4, 4), SparseMmaEngine::PackedInteger)
                }
                (64, ("int32", "int4" | "uint4", "int4" | "uint4", "int32"), _) => {
                    ((2, 2, 4), SparseMmaEngine::PackedInteger)
                }
                (128, ("int32", "int4" | "uint4", "int4" | "uint4", "int32"), _) => {
                    ((4, 4, 4), SparseMmaEngine::PackedInteger)
                }
                _ => return None,
            };
        Some(SparseMmaVariant {
            a_count,
            b_count,
            c_count,
            engine,
        })
    }
}
