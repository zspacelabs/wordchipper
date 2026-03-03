//! # Vocabulary Factories

use once_cell::sync::OnceCell;
use spin::RwLock;

use crate::{
    TokenType,
    UnifiedTokenVocab,
    WCError,
    WCResult,
    alloc::{
        format,
        string::String,
        sync::Arc,
    },
    prelude::*,
    pretrained::{
        vocab_description::VocabDescription,
        vocab_query::VocabQuery,
    },
    support::resources::ResourceLoader,
};

/// Global vocabulary factory.
static FACTORY: OnceCell<RwLock<VocabFactory>> = OnceCell::new();

/// Hook for registering vocabulary providers at runtime.
pub struct VocabProviderInventoryHook {
    /// Builder function to create a new vocabulary provider.
    pub builder: fn() -> Arc<dyn VocabProvider>,
}
inventory::collect!(VocabProviderInventoryHook);

impl VocabProviderInventoryHook {
    /// Create a new inventory hook.
    pub const fn new(builder: fn() -> Arc<dyn VocabProvider>) -> Self {
        Self { builder }
    }
}

/// Get the global vocabulary factory.
pub fn get_vocab_factory() -> &'static RwLock<VocabFactory> {
    FACTORY.get_or_init(|| RwLock::new(init_factory()))
}

fn init_factory() -> VocabFactory {
    let mut factory = VocabFactory::default();

    for hook in inventory::iter::<VocabProviderInventoryHook> {
        factory.register_provider((hook.builder)()).unwrap();
    }

    factory
}

/// Run a function with mutable access to the global vocabulary factory.
pub fn with_vocab_factory_mut<F, V>(func: &mut F) -> V
where
    F: FnMut(&mut VocabFactory) -> V,
{
    let mut guard = get_vocab_factory().write();
    let factory = &mut *guard;
    func(factory)
}

/// Run a function with access to the global vocabulary factory.
pub fn with_vocab_factory<F, V>(func: &mut F) -> V
where
    F: FnMut(&VocabFactory) -> V,
{
    let guard = get_vocab_factory().read();
    let factory = &*guard;
    func(factory)
}

/// List all known vocabularies across all loaders.
pub fn list_vocabs() -> Vec<VocabListing> {
    with_vocab_factory(&mut |f: &VocabFactory| f.list_vocabs())
}

/// Resolve a [`VocabListing`] by name.
pub fn resolve_vocab(name: &str) -> WCResult<VocabDescription> {
    with_vocab_factory(&mut |f: &VocabFactory| f.resolve_vocab(name))
}

/// Load a [`UnifiedTokenVocab`] by name.
///
/// ## Returns
/// * `Ok(LabeledVocab<u32>)` - on success.
/// * `Err(WCError::ResourceNotFound)` - if the vocabulary is not found.
/// * `Err(e)` - on any other error.
pub fn load_vocab(
    name: &str,
    loader: &mut dyn ResourceLoader,
) -> WCResult<LabeledVocab<u32>> {
    with_vocab_factory(&mut move |f: &VocabFactory| f.load_vocab(name, loader))
}

/// List the available pretrained models.
///
/// ## Arguments
/// * `aliases` - Whether to include all aliases or just the primary names.
pub fn list_models() -> Vec<String> {
    let mut res = Vec::new();
    for listing in list_vocabs() {
        for descr in &listing.vocabs {
            res.push(descr.id().to_string());
        }
    }
    res
}

/// A listing of known tokenizer.
#[derive(Debug, Clone)]
pub struct VocabListing {
    /// The id of the factory that produced the vocabularies.
    source: String,

    /// A description of the factory.
    description: String,

    /// Explicitly listed vocabularies.
    vocabs: Vec<VocabDescription>,
}

impl VocabListing {
    /// Build a new vocabulary listing.
    pub fn new(
        source: &str,
        description: &str,
        vocabs: Vec<VocabDescription>,
    ) -> Self {
        Self {
            source: source.to_string(),
            description: description.to_string(),
            vocabs,
        }
    }

    /// Get the id of the factory that produced the vocabularies.
    pub fn provider(&self) -> &str {
        &self.source
    }

    /// Get the description of the factory.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Get the explicit list of vocabularies.
    pub fn vocabs(&self) -> &[VocabDescription] {
        &self.vocabs
    }
}

/// Resolved vocabulary with its description and loaded vocabulary.
#[derive(Clone)]
pub struct LabeledVocab<T: TokenType> {
    description: VocabDescription,
    vocab: Arc<UnifiedTokenVocab<T>>,
}

impl<T: TokenType> LabeledVocab<T> {
    /// Build a new resolved vocabulary.
    pub fn new(
        description: VocabDescription,
        vocab: Arc<UnifiedTokenVocab<T>>,
    ) -> Self {
        Self { description, vocab }
    }

    /// Get the description of the vocabulary.
    pub fn description(&self) -> &VocabDescription {
        &self.description
    }

    /// Get the unified token vocabulary.
    pub fn vocab(&self) -> &Arc<UnifiedTokenVocab<T>> {
        &self.vocab
    }
}

/// A factory for searching for and loading.
pub trait VocabProvider: Sync + Send {
    /// The name of the factory.
    fn name(&self) -> String;

    /// Get an extended description of the factory.
    fn description(&self) -> String;

    /// Get a listing of known vocabularies.
    fn list_vocabs(&self) -> Vec<VocabDescription>;

    /// Resolve a vocabulary description.
    ///
    /// ## Returns
    /// * `Ok(description)` - on success.
    /// * `Err(WCError::ResourceNotFound)` - if the vocabulary is not found.
    /// * `Err(e)` - on any other error.
    fn resolve_vocab(
        &self,
        query: &VocabQuery,
    ) -> WCResult<VocabDescription> {
        for desc in self.list_vocabs() {
            if query.schema().is_some() && query.schema() != desc.id().schema() {
                continue;
            }
            if query.path().is_some() && query.path() != desc.id().path() {
                continue;
            }

            if query.name() == desc.id().name() {
                return Ok(desc);
            }
        }
        Err(WCError::ResourceNotFound(query.to_string()))
    }

    /// Load a vocabulary from a name.
    ///
    /// ## Returns
    /// * `Ok((desc, vocab))` - on success.
    /// * `Err(WCError::ResourceNotFound)` - the vocabulary is not found.
    /// * `Err(e)` - on any other error.
    fn load_vocab(
        &self,
        query: &VocabQuery,
        loader: &mut dyn ResourceLoader,
    ) -> WCResult<LabeledVocab<u32>>;
}

/// A factory for searching for and loading vocabularies.
#[derive(Default)]
pub struct VocabFactory {
    providers: Vec<Arc<dyn VocabProvider>>,
}

impl VocabFactory {
    /// Get a reference to the registered vocabulary providers.
    pub fn providers(&self) -> &[Arc<dyn VocabProvider>] {
        &self.providers
    }

    /// Find a provider by its id.
    pub fn find_provider(
        &self,
        id: &str,
    ) -> Option<&Arc<dyn VocabProvider>> {
        self.providers
            .iter()
            .find(|p| p.name().to_lowercase() == id.to_lowercase())
    }

    /// Register a new [`VocabProvider`].
    ///
    /// ## Returns
    /// * `Ok(())` - on success,
    /// * `Err(WCError::DuplicatedResource)` - if a provider with the same name
    ///   already exists
    pub fn register_provider(
        &mut self,
        provider: Arc<dyn VocabProvider>,
    ) -> WCResult<()> {
        let id = provider.name().to_lowercase();

        for existing in &self.providers {
            if id == existing.name().to_lowercase() {
                return Err(WCError::DuplicatedResource(format!(
                    "Vocabulary provider with id '{id}' already exists",
                )));
            }
        }
        self.providers.push(provider);
        Ok(())
    }

    /// Remove a [`VocabProvider`].
    ///
    /// ## Returns
    /// The removed resource, if any.
    pub fn remove_provider(
        &mut self,
        id: &str,
    ) -> Option<Arc<dyn VocabProvider>> {
        self.providers
            .iter()
            .position(|p| p.name() == id)
            .map(|i| self.providers.remove(i))
    }

    /// List all known vocabularies across all loaders.
    pub fn list_vocabs(&self) -> Vec<VocabListing> {
        let mut res = Vec::new();
        for provider in &self.providers {
            res.push(VocabListing {
                source: provider.name(),
                description: provider.description(),
                vocabs: provider.list_vocabs(),
            });
        }
        res
    }

    /// Resolve a [`VocabDescription`] by name.
    ///
    /// ## Returns
    /// * `Ok(description)` - on success.
    /// * `Err(WCError::ResourceNotFound)` - if the vocabulary is not found.
    /// * `Err(e)` - on any other error.
    pub fn resolve_vocab<Q>(
        &self,
        query: Q,
    ) -> WCResult<VocabDescription>
    where
        Q: Into<VocabQuery>,
    {
        let query = query.into();
        for provider in &self.providers {
            match provider.resolve_vocab(&query) {
                Ok(vocab) => return Ok(vocab),
                Err(WCError::ResourceNotFound(_)) => (),
                Err(err) => return Err(err),
            }
        }
        Err(WCError::ResourceNotFound(query.to_string()))
    }

    /// Load a [`UnifiedTokenVocab`] by name.
    ///
    /// ## Returns
    /// * `Ok(LabeledVocab<u32>)` - on success.
    /// * `Err(WCError::ResourceNotFound)` - if the vocabulary is not found.
    /// * `Err(e)` - on any other error.
    pub fn load_vocab<Q>(
        &self,
        query: Q,
        loader: &mut dyn ResourceLoader,
    ) -> WCResult<LabeledVocab<u32>>
    where
        Q: Into<VocabQuery>,
    {
        let query = query.into();
        for provider in &self.providers {
            match provider.load_vocab(&query, loader) {
                Ok(vocab) => return Ok(vocab),
                Err(WCError::ResourceNotFound(_)) => (),
                Err(err) => return Err(err),
            }
        }
        Err(WCError::ResourceNotFound(query.to_string()))
    }
}
