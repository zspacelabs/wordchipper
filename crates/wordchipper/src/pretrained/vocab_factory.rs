//! # Vocabulary Factories

use once_cell::sync::OnceCell;
use spin::RwLock;

use crate::{
    UnifiedTokenVocab,
    WCError,
    WCResult,
    alloc::{
        format,
        string::String,
        sync::Arc,
    },
    prelude::*,
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
/// * `Ok((desc, vocab))` - on success.
/// * `Err(WCError::ResourceNotFound)` - if the vocabulary is not found.
/// * `Err(e)` - on any other error.
pub fn load_vocab(
    name: &str,
    loader: &mut dyn ResourceLoader,
) -> WCResult<(VocabDescription, Arc<UnifiedTokenVocab<u32>>)> {
    with_vocab_factory(&mut move |f: &VocabFactory| f.load_vocab(name, loader))
}

/// List the available pretrained models.
///
/// ## Arguments
/// * `aliases` - Whether to include all aliases or just the primary names.
pub fn list_models() -> Vec<String> {
    let mut res = Vec::new();
    for listing in list_vocabs() {
        let source = listing.source;
        for descr in &listing.vocabs {
            let name = format!("{source}::{}", descr.id.clone());
            res.push(name);
        }
    }
    res
}

/// A description of a pretrained tokenizer.
#[derive(Debug, Clone)]
pub struct VocabDescription {
    /// The resolution id of the vocabulary.
    pub id: String,

    /// The cache context for the vocabulary.
    pub context: Vec<String>,

    /// A description of the vocabulary.
    pub description: String,
}

/// A listing of known tokenizer.
#[derive(Debug, Clone)]
pub struct VocabListing {
    /// The id of the factory that produced the vocabularies.
    pub source: String,

    /// A description of the factory.
    pub description: String,

    /// Explicitly listed vocabularies.
    pub vocabs: Vec<VocabDescription>,
}

/// A factory for searching for and loading.
pub trait VocabProvider: Sync + Send {
    /// The name of the factory.
    fn id(&self) -> String;

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
        name: &str,
    ) -> WCResult<VocabDescription> {
        for vocab in self.list_vocabs() {
            if vocab.id == name {
                return Ok(vocab);
            }
        }
        Err(WCError::ResourceNotFound(name.to_string()))
    }

    /// Load a vocabulary from a name.
    ///
    /// ## Returns
    /// * `Ok((desc, vocab))` - on success.
    /// * `Err(WCError::ResourceNotFound)` - the vocabulary is not found.
    /// * `Err(e)` - on any other error.
    fn load_vocab(
        &self,
        name: &str,
        loader: &mut dyn ResourceLoader,
    ) -> WCResult<(VocabDescription, Arc<UnifiedTokenVocab<u32>>)>;
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
            .find(|p| p.id().to_lowercase() == id.to_lowercase())
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
        let id = provider.id().to_lowercase();

        for existing in &self.providers {
            if id == existing.id().to_lowercase() {
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
            .position(|p| p.id() == id)
            .map(|i| self.providers.remove(i))
    }

    /// List all known vocabularies across all loaders.
    pub fn list_vocabs(&self) -> Vec<VocabListing> {
        let mut res = Vec::new();
        for provider in &self.providers {
            res.push(VocabListing {
                source: provider.id(),
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
    pub fn resolve_vocab(
        &self,
        name: &str,
    ) -> WCResult<VocabDescription> {
        if name.contains("::") {
            let (provider_name, vocab_name) = name.split_once("::").unwrap();

            if let Some(provider) = self.find_provider(provider_name) {
                return match provider.resolve_vocab(vocab_name) {
                    Ok(vocab) => Ok(vocab),
                    Err(WCError::ResourceNotFound(_)) => {
                        Err(WCError::ResourceNotFound(name.to_string()))
                    }
                    Err(err) => Err(err),
                };
            }
        } else {
            for provider in &self.providers {
                match provider.resolve_vocab(name) {
                    Ok(vocab) => return Ok(vocab),
                    Err(WCError::ResourceNotFound(_)) => (),
                    Err(err) => return Err(err),
                }
            }
        }
        Err(WCError::ResourceNotFound(name.to_string()))
    }

    /// Load a [`UnifiedTokenVocab`] by name.
    ///
    /// ## Returns
    /// * `Ok((desc, vocab))` - on success.
    /// * `Err(WCError::ResourceNotFound)` - if the vocabulary is not found.
    /// * `Err(e)` - on any other error.
    pub fn load_vocab(
        &self,
        name: &str,
        loader: &mut dyn ResourceLoader,
    ) -> WCResult<(VocabDescription, Arc<UnifiedTokenVocab<u32>>)> {
        if name.contains("::") {
            let (provider_name, vocab_name) = name.split_once("::").unwrap();

            if let Some(provider) = self.find_provider(provider_name) {
                return match provider.load_vocab(vocab_name, loader) {
                    Ok(vocab) => Ok(vocab),
                    Err(WCError::ResourceNotFound(_)) => {
                        Err(WCError::ResourceNotFound(name.to_string()))
                    }
                    Err(err) => Err(err),
                };
            }
        } else {
            for provider in &self.providers {
                match provider.load_vocab(name, loader) {
                    Ok(vocab) => return Ok(vocab),
                    Err(WCError::ResourceNotFound(_)) => (),
                    Err(err) => return Err(err),
                }
            }
        }
        Err(WCError::ResourceNotFound(name.to_string()))
    }
}
