use nekoengine::{Engine, WindowConfig};
use nekohazard_game::GamePlugin;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Layer;
use tracing_subscriber::Registry;
use tracing_subscriber::layer::SubscriberExt;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::android::EventLoopBuilderExtAndroid;

#[cfg(target_os = "android")]
#[unsafe(no_mangle)]
fn android_main(android_app: winit::platform::android::activity::AndroidApp) {
    let android_layer =
        tracing_android::layer("nekohazard").expect("Failed to create Android tracing layer");

    let filter = EnvFilter::new("info,winit=off,nekoengine=debug,nekohazard_android=debug");
    let subscriber = Registry::default().with(android_layer).with(filter);
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set global subscriber");

    std::panic::set_hook(Box::new(|info| tracing::error!("panic: {info}")));

    Engine::builder()
        .window(WindowConfig {
            title: "Nekohazard".to_string(),
            size: (800, 600),
            grab_cursor: false,
        })
        .add_plugin(GamePlugin)
        .run()?;
}
