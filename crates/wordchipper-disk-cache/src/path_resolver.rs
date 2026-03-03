//! # App Path Resolver
//!
//! Static library defaults for cache/data directory resolution.

use std::{
    env,
    path::{
        Path,
        PathBuf,
    },
};

use directories_next::ProjectDirs;

/// Static configuration for application path resolution.
pub struct PathResolver {
    /// The qualifier for [`ProjectDirs`].
    pub qualifier: &'static str,

    /// The organization for [`ProjectDirs`].
    pub organization: &'static str,

    /// The application for [`ProjectDirs`].
    pub application: &'static str,

    /// The resolution order for cache directories environment variables.
    pub cache_env_vars: &'static [&'static str],

    /// The resolution order for data directories environment variables.
    pub data_env_vars: &'static [&'static str],
}

impl PathResolver {
    /// Get the [`ProjectDirs`] for this config.
    pub fn project_dirs(&self) -> Option<ProjectDirs> {
        ProjectDirs::from(self.organization, self.application, self.qualifier)
    }

    /// Resolve the cache directory for this config.
    ///
    /// Resolution Order:
    /// 1. `path`, if present.
    /// 2. ``env[$VAR]`` for each `self.cache_env_vars`; in order.
    /// 3. `self.project_dirs().cache_dir()`, if present.
    /// 4. `None`
    ///
    /// ## Project Dirs Behavior
    ///
    /// |Platform | Value                                                                 | Example                                             |
    /// | ------- | --------------------------------------------------------------------- | --------------------------------------------------- |
    /// | Linux   | `$XDG_CACHE_HOME`/`_project_path_` or `$HOME`/.cache/`_project_path_` | /home/alice/.cache/barapp                           |
    /// | macOS   | `$HOME`/Library/Caches/`_project_path_`                               | /Users/Alice/Library/Caches/com.Foo-Corp.Bar-App    |
    /// | Windows | `{FOLDERID_LocalAppData}`\\`_project_path_`\\cache                    | C:\Users\Alice\AppData\Local\Foo Corp\Bar App\cache |
    pub fn resolve_cache_dir<P: AsRef<Path>>(
        &self,
        path: Option<P>,
    ) -> Option<PathBuf> {
        if let Some(path) = path.as_ref() {
            return Some(path.as_ref().to_path_buf());
        }

        for env_var in self.cache_env_vars {
            if let Ok(path) = env::var(env_var) {
                return Some(PathBuf::from(path));
            }
        }

        if let Some(pds) = self.project_dirs() {
            return Some(pds.cache_dir().to_path_buf());
        }

        None
    }

    /// Resolve the data directory for this config.
    ///
    /// Resolution Order:
    /// 1. `path`, if present.
    /// 2. ``env[$VAR]`` for each `self.data_env_vars`; in order.
    /// 3. `self.project_dirs().data_dirs()`, if present.
    /// 4. `None`
    ///
    /// ## Project Dirs Behavior
    ///
    /// |Platform | Value                                                                      | Example                                                       |
    /// | ------- | -------------------------------------------------------------------------- | ------------------------------------------------------------- |
    /// | Linux   | `$XDG_DATA_HOME`/`_project_path_` or `$HOME`/.local/share/`_project_path_` | /home/alice/.local/share/barapp                               |
    /// | macOS   | `$HOME`/Library/Application Support/`_project_path_`                       | /Users/Alice/Library/Application Support/com.Foo-Corp.Bar-App |
    /// | Windows | `{FOLDERID_LocalAppData}`\\`_project_path_`\\data                          | C:\Users\Alice\AppData\Local\Foo Corp\Bar App\data            |
    pub fn resolve_data_dir<P: AsRef<Path>>(
        &self,
        path: Option<P>,
    ) -> Option<PathBuf> {
        if let Some(path) = path.as_ref() {
            return Some(path.as_ref().to_path_buf());
        }

        for env_var in self.data_env_vars {
            if let Ok(path) = env::var(env_var) {
                return Some(PathBuf::from(path));
            }
        }

        if let Some(pds) = self.project_dirs() {
            return Some(pds.data_dir().to_path_buf());
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;

    const CACHE_ENV1: &str = "_APP_PATH_CACHE_ENV1";
    const CACHE_ENV2: &str = "_APP_PATH_CACHE_ENV2";
    const DATA_ENV1: &str = "_APP_PATH_DATA_ENV1";
    const DATA_ENV2: &str = "_APP_PATH_DATA_ENV2";

    const TEST_CONFIG: PathResolver = PathResolver {
        qualifier: "io",
        organization: "crates",
        application: "example",
        cache_env_vars: &[CACHE_ENV1, CACHE_ENV2],
        data_env_vars: &[DATA_ENV1, DATA_ENV2],
    };

    #[test]
    #[serial]
    fn test_resolve_dirs() {
        let pds = TEST_CONFIG
            .project_dirs()
            .expect("failed to get project dirs");

        let no_path: Option<PathBuf> = None;

        let user_cache_dir = PathBuf::from("/tmp/app_cache/cache");
        let user_data_dir = PathBuf::from("/tmp/app_cache/data");

        let env_cache_dir1 = PathBuf::from("/tmp/app_cache/env_cache.1");
        let env_cache_dir2 = PathBuf::from("/tmp/app_cache/env_cache.2");
        let env_data_dir1 = PathBuf::from("/tmp/app_cache/env_data.1");
        let env_data_dir2 = PathBuf::from("/tmp/app_cache/env_data.2");

        // No env vars
        unsafe {
            for v in TEST_CONFIG.cache_env_vars {
                env::remove_var(v);
            }
            for v in TEST_CONFIG.data_env_vars {
                env::remove_var(v);
            }
        }

        // User overrides.
        assert_eq!(
            TEST_CONFIG.resolve_cache_dir(Some(user_cache_dir.clone())),
            Some(user_cache_dir.clone()),
        );
        assert_eq!(
            TEST_CONFIG.resolve_data_dir(Some(user_data_dir.clone())),
            Some(user_data_dir.clone()),
        );

        // Resolution; use project dirs.
        assert_eq!(
            TEST_CONFIG.resolve_cache_dir(no_path.clone()),
            Some(pds.cache_dir().to_path_buf())
        );
        assert_eq!(
            TEST_CONFIG.resolve_data_dir(no_path.clone()),
            Some(pds.data_dir().to_path_buf())
        );

        // Lowest priority dirs.
        unsafe {
            env::set_var(CACHE_ENV2, env_cache_dir2.to_str().unwrap());
            env::set_var(DATA_ENV2, env_data_dir2.to_str().unwrap());
        }

        // User overrides.
        assert_eq!(
            TEST_CONFIG.resolve_cache_dir(Some(user_cache_dir.clone())),
            Some(user_cache_dir.clone()),
        );
        assert_eq!(
            TEST_CONFIG.resolve_data_dir(Some(user_data_dir.clone())),
            Some(user_data_dir.clone()),
        );

        // Resolution; use env vars.
        assert_eq!(
            TEST_CONFIG.resolve_cache_dir(no_path.clone()),
            Some(env_cache_dir2.clone())
        );
        assert_eq!(
            TEST_CONFIG.resolve_data_dir(no_path.clone()),
            Some(env_data_dir2.clone())
        );

        // Higher priority dirs.
        unsafe {
            env::set_var(CACHE_ENV1, env_cache_dir1.to_str().unwrap());
            env::set_var(DATA_ENV1, env_data_dir1.to_str().unwrap());
        }

        // User overrides.
        assert_eq!(
            TEST_CONFIG.resolve_cache_dir(Some(user_cache_dir.clone())),
            Some(user_cache_dir.clone()),
        );
        assert_eq!(
            TEST_CONFIG.resolve_data_dir(Some(user_data_dir.clone())),
            Some(user_data_dir.clone()),
        );

        // Resolution; use env vars.
        assert_eq!(
            TEST_CONFIG.resolve_cache_dir(no_path.clone()),
            Some(env_cache_dir1.clone())
        );
        assert_eq!(
            TEST_CONFIG.resolve_data_dir(no_path.clone()),
            Some(env_data_dir1.clone())
        );
    }
}
