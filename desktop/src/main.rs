use anyhow::Result;
use nekoengine::{Engine, WindowConfig};
use nekohazard_game::GamePlugin;
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

fn main() -> Result<()> {
    let filter_layer = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(tracing_tracy::TracyLayer::default())
        .with(fmt::layer())
        .with(filter_layer)
        .init();

    Engine::builder()
        .window(WindowConfig {
            title: "Nekohazard".to_string(),
            size: (800, 600),
            grab_cursor: true,
        })
        .add_plugin(GamePlugin)
        .run()?;

    Ok(())
}
