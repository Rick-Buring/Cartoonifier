mod cartoonify;
mod UI;

use UI::CartoonifyApp;
fn main() -> Result<(), eframe::Error> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Cartoonifyer",
        native_options,
        Box::new(|cc| Box::new(CartoonifyApp::new(cc))),
    )
}

