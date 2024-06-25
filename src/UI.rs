use eframe::egui::{ColorImage, TextureHandle};
use egui::{Slider, TextEdit};

use rfd::FileDialog;

use crate::cartoonify::Cartoonify;
use opencv::core::Mat;
use opencv::imgproc;
use opencv::prelude::MatTraitConst;
use opencv::prelude::MatTraitConstManual;
#[derive(Default)]
pub(crate) struct CartoonifyApp {
    sliders: [f32; 5],
    checkboxes: [bool; 5],
    cartoonify: Cartoonify,
}

impl CartoonifyApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self::default()
    }
}

impl eframe::App for CartoonifyApp {


    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label("Path:");
                    ui.add(TextEdit::singleline(&mut self.cartoonify.path));

                    ui.label("Blur Factor:");
                    ui.add(Slider::new(&mut self.cartoonify.blur_factor, 0..=100));

                    ui.label("Max Value:");
                    ui.add(Slider::new(&mut self.cartoonify.max_value, 0.0..=500.0));

                    ui.label("Block Size:");
                    ui.add(Slider::new(&mut self.cartoonify.block_size, 1..=50));

                    ui.label("C:");
                    ui.add(Slider::new(&mut self.cartoonify.c, 0.0..=10.0));

                    ui.label("Min Area:");
                    ui.add(Slider::new(&mut self.cartoonify.min_area, 0.0..=100.0));

                    ui.label("Min Length:");
                    ui.add(Slider::new(&mut self.cartoonify.min_length, 0.0..=100.0));

                    ui.label("Kernel X:");
                    ui.add(Slider::new(&mut self.cartoonify.kernel_x, 1..=10));

                    ui.label("Kernel Y:");
                    ui.add(Slider::new(&mut self.cartoonify.kernel_y, 1..=10));

                    ui.label("Iterations:");
                    ui.add(Slider::new(&mut self.cartoonify.iterations, 0..=10));

                    ui.label("Groups:");
                    ui.add(Slider::new(&mut self.cartoonify.groups, 0..=20));

                    ui.label("Attempts:");
                    ui.add(Slider::new(&mut self.cartoonify.attempts, 0..=20));

                    ui.label("Max Count:");
                    ui.add(Slider::new(&mut self.cartoonify.max_count, 0..=200));

                    ui.label("Epsilon:");
                    ui.add(Slider::new(&mut self.cartoonify.epsilon, 0.0..=1.0));

                    if ui.button("Load").clicked() {

                        self.load_images(ctx);
                    }
                    if ui.button("Update").clicked() {
                        self.update_cartoon_image();
                    }
                });
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {

                        let old_image = self.cartoonify.old_cartoon.clone();
                        match  old_image {
                            None => {
                                ui.label("OLD");
                            }
                            Some(im) => {
                                let new_width= 400;

                                let mut image = Mat::default();
                                let new_height = (im.rows() as f32 / im.cols() as f32 * new_width as f32) as i32;

                                imgproc::resize(&im, &mut image, opencv::core::Size::new(new_width, new_height), 0.0,0.0, imgproc::INTER_AREA).ok().unwrap();
                                let color_image = mat_to_color_image(&image);
                                let texture: TextureHandle = ui.ctx().load_texture("old_dynamic", color_image, Default::default());
                                ui.image(&texture);
                            }
                        }
                        ui.separator();

                        let original_image = self.cartoonify.original_image.clone();
                        match original_image {
                            None => {
                                ui.label("OG-img");
                            }
                            Some(im) => {
                                let new_width= 400;

                                let mut image = Mat::default();
                                let new_height = (im.rows() as f32 / im.cols() as f32 * new_width as f32) as i32;

                                imgproc::resize(&im, &mut image, opencv::core::Size::new(new_width, new_height), 0.0,0.0, imgproc::INTER_AREA).ok().unwrap();
                                let color_image = mat_to_color_image(&image);
                                let texture: TextureHandle = ui.ctx().load_texture("original_dynamic", color_image, Default::default());
                                ui.image(&texture);
                            }
                        }
                    });
                    ui.separator();
                    let original_image = self.cartoonify.cartoon.clone();
                    match  original_image {
                        None => {
                            ui.label("Cartoon");
                        }
                        Some(im) => {
                            let new_width= 800;

                            let mut image = Mat::default();
                            let new_height = (im.rows() as f32 / im.cols() as f32 * new_width as f32) as i32;

                            imgproc::resize(&im, &mut image, opencv::core::Size::new(new_width, new_height), 0.0,0.0, imgproc::INTER_AREA).ok().unwrap();
                            let color_image = mat_to_color_image(&image);
                            let texture: TextureHandle = ui.ctx().load_texture("Cartoon_dynamic", color_image, Default::default());
                            ui.image(&texture);
                        }
                    }

                });
            });
        });
    }
}
fn mat_to_color_image(mat: &Mat) -> ColorImage {
    let (width, height) = (mat.cols() as usize, mat.rows() as usize);

    // Convert BGR to RGB
    let mut rgb_image = Mat::default();
    imgproc::cvt_color(&mat, &mut rgb_image, imgproc::COLOR_BGR2RGB, 0).unwrap();

    // Convert Mat to Vec<u8>
    let size = (rgb_image.total() * rgb_image.elem_size().unwrap()) as usize;
    let rgb_bytes: Vec<u8> = rgb_image.data_bytes().unwrap().to_vec();

    // Convert Vec<u8> to ColorImage
    ColorImage::from_rgb([width, height], &rgb_bytes)
}

fn select_file() -> Option<String> {
    let file = FileDialog::new().pick_file();
    file.map(|path| path.to_string_lossy().to_string())
}

impl CartoonifyApp {
    fn load_images(&mut self, _ctx: &egui::Context) {
        // Implement loading images from files or other sources
        match select_file() {
            Some(path) => {
                self.cartoonify.set_path(path);
                self.cartoonify.load_image();
            }
            _ => {}
        }
    }

    fn update_cartoon_image(&mut self) {
        self.cartoonify.update(|cartoonify| cartoonify.process());
    }
}