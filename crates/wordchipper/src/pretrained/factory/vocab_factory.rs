//! # Vocabulary Factories

use once_cell::sync::OnceCell;
use spin::RwLock;

use crate::{
    WCError,
    WCResult,
    alloc::{
        format,
        string::String,
        sync::Arc,
    },
    prelude::*,
    pretrained::factory::{
        vocab_description::{
            LabeledVocab,
            VocabDescription,
            VocabListing,
        },
        vocab_provider::VocabProvider,
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

/// Load a [`LabeledVocab`] by name.
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
        for descr in listing.vocabs() {
            res.push(descr.id().to_string());
        }
    }
    res
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
            let listing = VocabListing::new(
                &provider.name(),
                &provider.description(),
                provider.list_vocabs(),
            );
            res.push(listing);
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
