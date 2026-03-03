//! # Remote Resource Tools

use crate::alloc::{
    string::{
        String,
        ToString,
    },
    vec::Vec,
};

/// A resource with a constant URL and optional hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstUrlResource {
    /// The URL associated with this resource.
    pub urls: &'static [&'static str],

    /// The hash associated with this resource, if available.
    pub hash: Option<&'static str>,
}

/// A resource with a list of URLs and optional hash.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UrlResource {
    /// The URLs associated with this resource.
    pub urls: Vec<String>,

    /// The hash associated with this resource, if available.
    pub hash: Option<String>,
}

impl From<ConstUrlResource> for UrlResource {
    fn from(resource: ConstUrlResource) -> Self {
        UrlResource {
            urls: resource.urls.iter().map(|s| s.to_string()).collect(),
            hash: resource.hash.map(|s| s.to_string()),
        }
    }
}

/// A keyed resource.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConstKeyedResource {
    /// The key associated with this resource.
    ///
    /// This is used in place of a URI for internal caching
    /// and fetch unification.
    pub key: &'static [&'static str],

    /// The resource associated with this key.
    pub resource: ConstUrlResource,
}

/// A keyed resource.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KeyedResource {
    /// The key associated with this resource.
    ///
    /// This is used in place of a URI for internal caching
    /// and fetch unification.
    pub key: Vec<String>,

    /// The resource associated with this key.
    pub resource: UrlResource,
}

impl From<ConstKeyedResource> for KeyedResource {
    fn from(resource: ConstKeyedResource) -> Self {
        KeyedResource {
            key: resource.key.iter().map(|s| s.to_string()).collect(),
            resource: resource.resource.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::{
        string::ToString,
        vec,
    };

    #[test]
    fn test_keyed_resource() {
        let cres = ConstKeyedResource {
            key: &["test_key"],
            resource: ConstUrlResource {
                urls: &["test_url"],
                hash: Some("test_hash"),
            },
        };

        let res: KeyedResource = cres.into();
        assert_eq!(res.key, vec!["test_key".to_string()]);
        assert_eq!(res.resource.urls, vec!["test_url".to_string()]);
        assert_eq!(res.resource.hash, Some("test_hash".to_string()));
    }
}
