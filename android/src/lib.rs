use nekoengine::{Engine, WindowConfig};
use nekohazard_game::GamePlugin;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Layer;
use tracing_subscriber::Registry;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::android::EventLoopBuilderExtAndroid;

#[cfg(target_os = "android")]
#[unsafe(no_mangle)]
fn android_main(android_app: winit::platform::android::activity::AndroidApp) {
    let android_layer =
        tracing_android::layer("nekohazard").expect("Failed to create Android tracing layer");

    let filter = EnvFilter::new("info,winit=off,nekoengine=debug,nekohazard_android=debug");
    tracing_subscriber::registry()
        .with(android_layer)
        .with(filter)
        .try_init();

    std::panic::set_hook(Box::new(|info| tracing::error!("panic: {info}")));

    let event_loop = EventLoop::builder()
        .with_android_app(android_app)
        .build()
        .unwrap();

    Engine::builder()
        .window(WindowConfig {
            title: "Nekohazard".to_string(),
            size: (800, 600),
            grab_cursor: false,
        })
        .add_plugin(GamePlugin)
        .run_event_loop(event_loop)
        .expect("Failed to run the engine!");
}
