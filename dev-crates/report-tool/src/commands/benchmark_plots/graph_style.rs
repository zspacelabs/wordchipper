#[derive(Debug, Clone, Copy)]
pub struct GraphStyleOptions {
    pub shape: (u32, u32),
    pub size: u32,
    pub line_width: u32,
}

impl Default for GraphStyleOptions {
    fn default() -> Self {
        Self {
            shape: (800, 900),
            size: 10,
            line_width: 6,
        }
    }
}
