//! # Public `OpenAI` Resources

use crate::support::resources::{
    ConstKeyedResource,
    ConstUrlResource,
};

/// Cache-keyed GPT-2 `DataGym` "vocab.bpe" vocabulary resource.
pub const OA_GPT2_VOCAB_BPE_KEYED_RESOURCE: ConstKeyedResource = ConstKeyedResource {
    key: &["openai", "gpt2"],
    resource: OA_GPT2_DATAGYM_VOCAB_BPE_RESOURCE,
};

/// Cache-keyed GPT-2 `DataGym` "encoder.json" encoder resource.
pub const OA_GPT2_ENCODER_JSON_KEYED_RESOURCE: ConstKeyedResource = ConstKeyedResource {
    key: &["openai", "gpt2"],
    resource: OA_GPT2_DATAGYM_ENCODER_JSON_RESOURCE,
};

/// The GPT-2 Data Gym "vocab.bpe" vocabulary resource.
pub const OA_GPT2_DATAGYM_VOCAB_BPE_RESOURCE: ConstUrlResource = ConstUrlResource {
    urls: &["https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe"],
    hash: Some("1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"),
};

/// The GPT-2 Data Gym "encoder.json" vocabulary resource.
pub const OA_GPT2_DATAGYM_ENCODER_JSON_RESOURCE: ConstUrlResource = ConstUrlResource {
    urls: &["https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json"],
    hash: Some("196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"),
};

/// The "`r50k_base.tiktoken`" vocabulary resource.
pub const OA_R50K_BASE_TIKTOKEN_RESOURCE: ConstUrlResource = ConstUrlResource {
    urls: &["https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken"],
    hash: Some("306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930"),
};

/// The "`p50k_base.tiktoken`" vocabulary resource.
pub const OA_P50K_BASE_TIKTOKEN_RESOURCE: ConstUrlResource = ConstUrlResource {
    urls: &["https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken"],
    hash: Some("94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069"),
};

/// The "`cl100k_base.tiktoken`" vocabulary resource.
pub const OA_CL100K_BASE_TIKTOKEN_RESOURCE: ConstUrlResource = ConstUrlResource {
    urls: &["https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"],
    hash: Some("223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"),
};

/// The "`o200k_base.tiktoken`" vocabulary resource.
pub const OA_O200K_BASE_TIKTOKEN_RESOURCE: ConstUrlResource = ConstUrlResource {
    urls: &["https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"],
    hash: Some("446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"),
};
