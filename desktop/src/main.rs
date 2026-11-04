use anyhow::Result;
use nekoengine::{Engine, WindowConfig};
use nekohazard_game::GamePlugin;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    let filter_layer = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_env_filter(filter_layer)
        .init();

    Engine::builder()
        .window(WindowConfig {
            title: "Nekohazard".to_string(),
            size: (1920, 1080),
            grab_cursor: true,
        })
        .add_plugin(GamePlugin)
        .run()?;

    Ok(())
}
