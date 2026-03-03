use core::{
    fmt::Display,
    str::FromStr,
};

use crate::{
    WCError,
    prelude::*,
};

/// A lookup query.
#[derive(Debug, Clone, PartialEq, Hash)]
pub struct VocabQuery {
    schema: Option<String>,
    path: Option<String>,
    name: String,
}

impl From<&str> for VocabQuery {
    fn from(s: &str) -> Self {
        Self::from_str(s).unwrap()
    }
}

impl FromStr for VocabQuery {
    type Err = WCError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut buf = s;

        let mut schema: Option<String> = None;
        let mut path: Option<String> = None;

        if s.contains(":") {
            let (sch, rem) = buf.split_once(":").unwrap();
            schema = Some(sch.to_string());
            buf = rem;
        }

        if s.contains("/") {
            let last = buf.rfind("/").unwrap();
            let grp = &buf[..last];
            path = Some(grp.to_string());
            buf = &buf[last + 1..];
        }

        let name = buf.to_string();

        Ok(VocabQuery { schema, path, name })
    }
}

impl Display for VocabQuery {
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        if let Some(schema) = &self.schema {
            write!(f, "{}:", schema)?;
        }
        if let Some(path) = &self.path {
            write!(f, "{}/", path)?;
        }
        write!(f, "{}", self.name)
    }
}

impl VocabQuery {
    /// Build a new query from structure.
    pub fn new(
        schema: Option<&str>,
        path: Option<&str>,
        name: &str,
    ) -> Self {
        Self {
            schema: schema.map(|s| s.to_string()),
            path: path.map(|s| s.to_string()),
            name: name.to_string(),
        }
    }

    /// Get the schema.
    pub fn schema(&self) -> Option<&str> {
        self.schema.as_deref()
    }

    /// Set the schema.
    pub fn set_schema(
        &mut self,
        schema: Option<&str>,
    ) {
        self.schema = schema.map(|s| s.to_string());
    }

    /// Set the schema.
    pub fn with_schema(
        self,
        schema: Option<&str>,
    ) -> Self {
        Self {
            schema: schema.map(|s| s.to_string()),
            ..self
        }
    }

    /// Get the path.
    pub fn path(&self) -> Option<&str> {
        self.path.as_deref()
    }

    /// Set the path.
    pub fn set_path(
        &mut self,
        path: Option<&str>,
    ) {
        self.path = path.map(|s| s.to_string());
    }

    /// Set the path.
    pub fn with_path(
        self,
        path: Option<&str>,
    ) -> Self {
        Self {
            path: path.map(|s| s.to_string()),
            ..self
        }
    }

    /// Get the name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Set the name of the vocabulary.
    ///
    /// If a structured value is passed, it will be parsed.
    pub fn set_name(
        &mut self,
        name: &str,
    ) {
        let q = Self::from_str(name).unwrap();

        if q.schema.is_some() {
            self.set_schema(q.schema.as_deref());
        }
        if q.path.is_some() {
            self.set_path(q.path.as_deref());
        }
        self.name = q.name;
    }

    /// Set the name of the vocabulary.
    ///
    /// If a structured value is passed, it will be parsed.
    pub fn with_name(
        mut self,
        name: &str,
    ) -> Self {
        self.set_name(name);
        self
    }
}

#[cfg(test)]
mod tests {
    use core::str::FromStr;

    use crate::{
        prelude::*,
        pretrained::vocab_query::VocabQuery,
    };

    #[test]
    fn test_vocab_query_to_from_str() {
        let q = VocabQuery::from_str("vocab_name").unwrap();
        assert_eq!(q.to_string(), "vocab_name");
        assert_eq!(q, VocabQuery::new(None, None, "vocab_name"));

        let q = VocabQuery::from_str("foo/bar/vocab_name").unwrap();
        assert_eq!(q.to_string(), "foo/bar/vocab_name");
        assert_eq!(q, VocabQuery::new(None, Some("foo/bar"), "vocab_name"));

        let q = VocabQuery::from_str("xyz:foo/bar/vocab_name").unwrap();
        assert_eq!(q.to_string(), "xyz:foo/bar/vocab_name");
        assert_eq!(
            q,
            VocabQuery::new(Some("xyz"), Some("foo/bar"), "vocab_name")
        );
    }
    #[test]
    fn test_vocab_query_with_schema() {
        let query = VocabQuery::new(None, None, "vocab_name").with_schema(Some("provider"));
        assert_eq!(query.schema(), Some("provider"));
        assert_eq!(query.name(), "vocab_name");
    }

    #[test]
    fn test_vocab_query_with_path() {
        let query = VocabQuery::new(None, None, "vocab_name").with_path(Some("path_name"));
        assert_eq!(query.path(), Some("path_name"));
        assert_eq!(query.name(), "vocab_name");
    }

    #[test]
    fn test_vocab_query_set_name() {
        let mut query = VocabQuery::new(Some("old"), Some("old_path"), "old_name");
        query.set_name("new_provider:new_path/new_name");
        assert_eq!(query.schema(), Some("new_provider"));
        assert_eq!(query.path(), Some("new_path"));
        assert_eq!(query.name(), "new_name");
    }

    #[test]
    fn test_vocab_query_set_name_partial() {
        let mut query = VocabQuery::new(Some("schema"), Some("path"), "name");
        query.set_name("new_name");
        assert_eq!(query.schema(), Some("schema"));
        assert_eq!(query.path(), Some("path"));
        assert_eq!(query.name(), "new_name");
    }

    #[test]
    fn test_vocab_query_with_name() {
        let query = VocabQuery::new(Some("old"), Some("old_path"), "old_name")
            .with_name("new_provider:new_path/new_name");
        assert_eq!(query.schema(), Some("new_provider"));
        assert_eq!(query.path(), Some("new_path"));
        assert_eq!(query.name(), "new_name");
    }
}
