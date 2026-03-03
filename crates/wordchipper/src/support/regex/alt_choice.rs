//! Exact Match Union Patterns

use crate::{
    alloc::{
        format,
        string::ToString,
        vec::Vec,
    },
    support::regex::regex_pattern::RegexPattern,
};

/// Create a union pattern of exact matches.
///
/// This will always be a [`RegexPattern::Basic`] variant.
///
/// ## Arguments
/// * `alts` - A slice of alternatives to union.
///
/// ## Returns
/// A new `RegexWrapperPattern::Basic` containing the union pattern.
pub fn alternate_choice_regex_pattern<S: AsRef<str>>(alts: &[S]) -> RegexPattern {
    let parts = alts
        .iter()
        .map(|s| fancy_regex::escape(s.as_ref()).to_string())
        .collect::<Vec<_>>();

    // turns out, 'fancy_regex' is ... faster?
    RegexPattern::Fancy(format!("({})", parts.join("|")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        alloc::vec,
        support::regex::RegexWrapper,
    };

    #[test]
    fn test_fixed_alternative_list() {
        let alternatives = ["apple", "[x]", "boat"];

        let pattern = alternate_choice_regex_pattern(&alternatives);
        assert_eq!(pattern.as_str(), r"(apple|\[x\]|boat)");

        let re: RegexWrapper = alternate_choice_regex_pattern(&alternatives)
            .compile()
            .unwrap();

        let text = "apple 123 [x] xyz boat";
        assert_eq!(re.find_iter(text).count(), 3);

        assert_eq!(
            re.find_iter(text).map(|m| m.range()).collect::<Vec<_>>(),
            vec![0..5, 10..13, 18..22]
        );
    }
}
