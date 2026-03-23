use humansize::FormatSizeOptions;

/// Format bytes per second (bps) into a human-readable string.
pub fn format_bps(bps: f64) -> String {
    let human_opts = FormatSizeOptions::from(humansize::BINARY).decimal_places(1);
    format!("{}/s", humansize::format_size(bps as u64, human_opts))
}
