//! `OpenAI` Tokenizer Special Tokens.

use crate::{
    TokenType,
    alloc::{
        string::{
            String,
            ToString,
        },
        vec::Vec,
    },
    declare_carrot_special,
    vocab::utility::{
        ToTokenList,
        format_reserved_carrot,
    },
};

declare_carrot_special!(
    (STARTOFTEXT, "startoftext"),
    (ENDOFTEXT, "endoftext"),
    (ENDOFPROMPT, "endofprompt"),
    (FIM_PREFIX, "fim_prefix"),
    (FIM_MIDDLE, "fim_middle"),
    (FIM_SUFFIX, "fim_suffix"),
    (RETURN, "return"),
    (CONSTRAIN, "constrain"),
    (CHANNEL, "channel"),
    (START, "start"),
    (END, "end"),
    (MESSAGE, "message"),
    (CALL, "call"),
);

/// The "r50k" special tokens.
pub const OA_R50K_SPECIAL_TOKENS: &[(&str, usize)] = &[(ENDOFTEXT, 50256)];

/// The "r50k" special tokens.
pub fn oa_r50k_base_special_tokens<T: TokenType>() -> Vec<(String, T)> {
    OA_R50K_SPECIAL_TOKENS.to_token_list::<T>()
}

/// The "p50k base" special tokens.
pub const OA_P50K_BASE_SPECIAL_TOKENS: &[(&str, usize)] = &[(ENDOFTEXT, 50256)];

/// The "p50k base" special tokens.
pub fn oa_p50k_base_special_tokens<T: TokenType>() -> Vec<(String, T)> {
    OA_P50K_BASE_SPECIAL_TOKENS.to_token_list::<T>()
}

/// The "p50k edit" special tokens.
pub const OA_P50K_EDIT_SPECIAL_TOKENS: &[(&str, usize)] = &[
    (ENDOFTEXT, 50256),
    (FIM_PREFIX, 50281),
    (FIM_MIDDLE, 50282),
    (FIM_SUFFIX, 50283),
];

/// The "p50k edit" special tokens.
pub fn oa_p50k_edit_special_tokens<T: TokenType>() -> Vec<(String, T)> {
    OA_P50K_EDIT_SPECIAL_TOKENS.to_token_list::<T>()
}

/// The "cl100k" special tokens.
pub const OA_CL100K_EDIT_SPECIAL_TOKENS: &[(&str, usize)] = &[
    (ENDOFTEXT, 100257),
    (FIM_PREFIX, 100258),
    (FIM_MIDDLE, 100259),
    (FIM_SUFFIX, 100260),
    (ENDOFPROMPT, 100276),
];

/// The "cl100k" special tokens.
pub fn oa_cl100k_edit_special_tokens<T: TokenType>() -> Vec<(String, T)> {
    OA_CL100K_EDIT_SPECIAL_TOKENS.to_token_list::<T>()
}

/// The "o200k base" special tokens.
pub const OA_O200K_BASE_SPECIAL_TOKENS: &[(&str, usize)] =
    &[(ENDOFTEXT, 199999), (ENDOFPROMPT, 200018)];

/// The "o200k base" special tokens.
pub fn oa_o200k_base_special_tokens<T: TokenType>() -> Vec<(String, T)> {
    OA_O200K_BASE_SPECIAL_TOKENS.to_token_list::<T>()
}

/// The "o200k harmony" special tokens.
pub const OA_O200K_HARMONY_NAMED_SPECIAL_TOKENS: &[(&str, usize)] = &[
    (STARTOFTEXT, 199998),
    (ENDOFTEXT, 199999),
    (ENDOFPROMPT, 200018),
    (RETURN, 200002),
    (CONSTRAIN, 200003),
    (CHANNEL, 200005),
    (START, 200006),
    (END, 200007),
    (MESSAGE, 200008),
    (CALL, 200012),
];

/// The "o200k harmony" named special tokens; excluding reserved tokens.
pub fn oa_o200k_harmony_named_special_tokens<T: TokenType>() -> Vec<(String, T)> {
    OA_O200K_HARMONY_NAMED_SPECIAL_TOKENS.to_token_list::<T>()
}

/// Generate the "`o200k_harmony`" reserved tokens.
pub fn oa_o200k_harmony_reserved_tokens<T: TokenType>() -> Vec<(String, T)> {
    let mut specials: Vec<(String, usize)> = Vec::with_capacity(6 + (201088 - 200013));

    let mut reserve = |val| {
        specials.push((format_reserved_carrot(val), val));
    };

    reserve(200000);
    reserve(200001);
    reserve(200004);
    reserve(200009);
    reserve(200010);
    reserve(200011);

    for val in 200013..201088 {
        reserve(val);
    }

    specials
        .iter()
        .map(|(s, t)| (s.to_string(), T::from_usize(*t).unwrap()))
        .collect()
}

/// The GPT-5 "o200k harmony" special tokens.
///
/// Generated due to the large number of reserved tokens.
pub fn oa_o200k_harmony_special_tokens<T: TokenType>() -> Vec<(String, T)> {
    let mut specials = oa_o200k_harmony_named_special_tokens();

    specials.extend(oa_o200k_harmony_reserved_tokens());

    specials
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        alloc::{
            string::ToString,
            vec,
            vec::Vec,
        },
        vocab::utility::format_reserved_carrot,
    };

    #[test]
    fn test_oa_gpt2_r50k_specials() {
        assert_eq!(
            oa_r50k_base_special_tokens::<usize>(),
            vec![("<|endoftext|>".to_string(), 50256),]
        );
    }

    #[test]
    fn test_oa_gpt2_p50k_base_specials() {
        assert_eq!(
            oa_p50k_base_special_tokens::<usize>(),
            vec![("<|endoftext|>".to_string(), 50256),]
        );
    }

    #[test]
    fn test_oa_gpt2_p50k_edit_specials() {
        assert_eq!(
            oa_p50k_edit_special_tokens::<usize>(),
            vec![
                ("<|endoftext|>".to_string(), 50256),
                ("<|fim_prefix|>".to_string(), 50281),
                ("<|fim_middle|>".to_string(), 50282),
                ("<|fim_suffix|>".to_string(), 50283),
            ]
        );
    }

    #[test]
    fn test_oa_gpt3_cl100k_edit_specials() {
        assert_eq!(
            oa_cl100k_edit_special_tokens::<usize>(),
            vec![
                ("<|endoftext|>".to_string(), 100257),
                ("<|fim_prefix|>".to_string(), 100258),
                ("<|fim_middle|>".to_string(), 100259),
                ("<|fim_suffix|>".to_string(), 100260),
                ("<|endofprompt|>".to_string(), 100276),
            ]
        );
    }

    #[test]
    fn test_oa_gpt5_o200k_base_specials() {
        assert_eq!(
            oa_o200k_base_special_tokens::<usize>(),
            vec![
                ("<|endoftext|>".to_string(), 199999),
                ("<|endofprompt|>".to_string(), 200018)
            ]
        )
    }

    #[test]
    fn test_oa_gpt5_o200k_harmony_specials() {
        let mut expected = vec![
            ("<|reserved_200000|>".to_string(), 200000),
            ("<|reserved_200001|>".to_string(), 200001),
            ("<|reserved_200004|>".to_string(), 200004),
            ("<|reserved_200009|>".to_string(), 200009),
            ("<|reserved_200010|>".to_string(), 200010),
            ("<|reserved_200011|>".to_string(), 200011),
        ];
        (200013..201088).for_each(|i| expected.push((format_reserved_carrot(i), i)));

        let reserved = oa_o200k_harmony_reserved_tokens();
        assert_eq!(&reserved, &expected);

        let named = oa_o200k_harmony_named_special_tokens();
        assert_eq!(
            &named,
            &vec![
                ("<|startoftext|>".to_string(), 199998),
                ("<|endoftext|>".to_string(), 199999),
                ("<|endofprompt|>".to_string(), 200018),
                ("<|return|>".to_string(), 200002),
                ("<|constrain|>".to_string(), 200003),
                ("<|channel|>".to_string(), 200005),
                ("<|start|>".to_string(), 200006),
                ("<|end|>".to_string(), 200007),
                ("<|message|>".to_string(), 200008),
                ("<|call|>".to_string(), 200012),
            ]
        );

        let expected = named
            .iter()
            .chain(reserved.iter())
            .cloned()
            .collect::<Vec<_>>();

        assert_eq!(oa_o200k_harmony_special_tokens(), expected);
    }
}
