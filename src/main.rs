use eframe::egui;
use eframe::egui::{ColorImage, TextureHandle};
use opencv::{
    core::Mat,
    imgcodecs::{imread, IMREAD_COLOR},
    imgproc::{self, AdaptiveThresholdTypes, ThresholdTypes},
    core::{MatTrait, MatTraitConst, Scalar, TermCriteria_EPS, TermCriteria_MAX_ITER},
    prelude::*
};

use rfd::FileDialog;

fn main() -> Result<(), eframe::Error> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Cartoonifyer",
        native_options,
        Box::new(|cc| Box::new(CartoonifyApp::new(cc))),
    )
}

#[derive(Default)]
struct CartoonifyApp {
    sliders: [f32; 5],
    checkboxes: [bool; 5],
    cartoonify: Cartoonify,
}

impl CartoonifyApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self::default()
    }
}

impl eframe::App for CartoonifyApp {


    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    for i in 0..5 {
                        ui.horizontal(|ui| {
                            ui.add(egui::Slider::new(&mut self.sliders[i], 0.0..=1.0));
                        });
                    }
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
                self.cartoonify.path = path;
                self.cartoonify.load_image();
            }
            _ => {}
        }
    }

    fn update_cartoon_image(&mut self) {
        self.cartoonify.update(|cartoonify| cartoonify.process());
    }
}


struct Cartoonify {
    original_image: Option<Mat>,
    old_cartoon: Option<Mat>,
    cartoon: Option<Mat>,

    //LoadImage
    path: String,
    //medianBlur
    blur_factor: i32,
    //adaptive blur
    max_value: f64,
    block_size: i32,
    c: f64,
    //Contours
    min_area: f64,
    min_length: f64,
    //dilation
    kernel_x: i32,
    kernel_y: i32,
    iterations: i32,
    //quantizing
    groups: i32,
    attempts: i32,
    max_count: i32,
    epsilon: f64,

    gray_blur: Mat,
    adaptive_threshold: Mat,
    contour_image: Mat,

    outline_image: Mat,
    quantize_image: Mat,
    combined_image: Mat
}

impl Default for Cartoonify{
    fn default() -> Self{
        Cartoonify{
            original_image: None,
            old_cartoon: None,
            cartoon: None,


            path: "C:\\Users\\rick\\Pictures\\rust-cartoon-car3.jpg".to_string(),
            blur_factor: 5,

            max_value: 255.0,
            block_size: 5,
            c: 5.0,
            min_area: 5.0,
            min_length: 50.0,
            kernel_x: 3,
            kernel_y: 3,
            iterations: 1,
            groups: 8,
            attempts: 10,
            max_count: 100,
            epsilon: 0.2,
            gray_blur: Default::default(),

            adaptive_threshold: Default::default(),
            contour_image: Default::default(),
            outline_image: Default::default(),
            quantize_image: Default::default(),
            combined_image: Default::default(),
        }
    }
}

impl Cartoonify{
    fn load_image(&mut self) -> Option<Mat> {
        self.original_image = Some(imread(self.path.as_str(), IMREAD_COLOR).ok()?);
        self.original_image.clone()
    }
    fn gray_and_blur(&mut self) -> Option<Mat> {
        let mut gray = Mat::default();

        imgproc::cvt_color(&self.original_image.as_ref()?, &mut gray, imgproc::COLOR_BGR2GRAY, 0).ok();
        imgproc::median_blur(&gray, &mut self.gray_blur, self.blur_factor).ok();

        Some(self.gray_blur.clone())
    }

    fn adaptive_threshold(&mut self) -> Option<Mat> {
        imgproc::adaptive_threshold(
            &self.gray_blur,
            &mut self.adaptive_threshold,
            self.max_value,
            i32::from(AdaptiveThresholdTypes::ADAPTIVE_THRESH_MEAN_C),
            i32::from(ThresholdTypes::THRESH_BINARY),
            self.block_size,
            self.c
        ).ok();

        Some(self.adaptive_threshold.clone())
    }

    fn draw_contours(&mut self) -> Option<Mat> {
        let mut contours =
            opencv::core::Vector::<opencv::core::Vector<opencv::core::Point>>::new();
        imgproc::find_contours(
            &self.adaptive_threshold,
            &mut contours,
            imgproc::RETR_TREE,
            imgproc::CHAIN_APPROX_SIMPLE,
            opencv::core::Point::default()
        ).ok();

        let mut filtered_contours =
            opencv::core::Vector::<opencv::core::Vector<opencv::core::Point>>::new();
        for contour in contours {
            if imgproc::contour_area(&contour, false).ok()? >= self.min_area &&
                imgproc::arc_length(&contour, false).ok()? >= self.min_length {
                filtered_contours.push(contour);
            }
        }

        self.contour_image = Mat::zeros_size(
            self.adaptive_threshold.size().ok()?,
            self.adaptive_threshold.typ()
        ).ok()?.to_mat().ok()?;

        imgproc::draw_contours(
            &mut self.contour_image,
            &filtered_contours,
            -1,
            opencv::core::Scalar::new(255.0, 0.0, 0.0,0.0),
            1,
            imgproc::LINE_8,
            &opencv::core::no_array(),
            i32::MAX,
            opencv::core::Point::default()
        ).ok()?;

        Some(self.contour_image.clone())
    }

    fn create_outline(&mut self) -> Option<Mat> {
        let kernel = Mat::ones_size(
            opencv::core::Size::new(self.kernel_x, self.kernel_y),
            opencv::core::CV_8U
        ).ok()?.to_mat().ok()?;

        let mut outline = Mat::default();

        imgproc::dilate(&self.contour_image,
                                &mut outline,
                                &kernel,
                                opencv::core::Point::default(),
                                self.iterations,
                                opencv::core::BORDER_CONSTANT,
                                imgproc::morphology_default_border_value().unwrap()
        ).ok()?;

        let mut inverted = Mat::default();
        opencv::core::bitwise_not(&outline, &mut inverted, &opencv::core::no_array()).ok()?;

        let mut bgr = Mat::default();
        imgproc::cvt_color(&inverted, &mut bgr, opencv::imgproc::COLOR_GRAY2BGR, 0).ok()?;

        imgproc::gaussian_blur(
            &bgr,
            &mut self.outline_image,
            opencv::core::Size::new(5, 5),
            1.0,
            0.0,
            opencv::core::BORDER_DEFAULT
        ).ok()?;


        Some(self.outline_image.clone())
    }

    fn quantize(& mut self) -> Option<Mat> {
        let original_img = self.original_image.as_ref()?;
        // Reshape the image to a 2D array of pixels
        let mut data =
            Mat::new_rows_cols_with_default(original_img.rows() * original_img.cols(), 3, opencv::core::CV_32F, Scalar::all(0.0)).ok()?;
        for r in 0..original_img.rows() {
            for c in 0..original_img.cols() {
                for k in 0..3 {
                    let val = original_img.at_2d::<opencv::core::Vec3b>(r, c).ok()?.0[k];
                    *data.at_2d_mut::<f32>(r * original_img.cols() + c, k as i32).ok()? = val as f32;
                }
            }
        }

        // Apply K-means clustering
        let mut labels = Mat::default();
        let mut centers = Mat::default();
        let term_criteria = opencv::core::TermCriteria::new(
            TermCriteria_EPS + TermCriteria_MAX_ITER,
            100,
            0.2,
        ).ok()?;
        opencv::core::kmeans(
            &data,
            12,
            &mut labels,
            term_criteria,
            10,
            opencv::core::KMEANS_RANDOM_CENTERS,
            &mut centers,
        ).ok()?;

        // Map the labels to the centers
        let mut quantized_img = Mat::zeros(original_img.rows(), original_img.cols(), original_img.typ()).ok()?.to_mat().ok()?;
        for r in 0..original_img.rows() {
            for c in 0..original_img.cols() {
                let cluster_idx = *labels.at::<i32>(r * original_img.cols() + c).ok()?;
                for k in 0..3 {
                    let center_val = *centers.at_2d::<f32>(cluster_idx, k).ok()?;
                    quantized_img.at_2d_mut::<opencv::core::Vec3b>(r, c).ok()?.0[k as usize] = center_val as u8;
                }
            }
        }
        self.quantize_image = quantized_img;
        Some(self.quantize_image.clone())
    }

    fn cartoonify(& mut self) -> Option<Mat> {
         opencv::core::bitwise_and(
             &self.quantize_image,
             &self.outline_image,
             &mut self.combined_image,
             &opencv::core::no_array()
         ).ok()?;

        Some(self.combined_image.clone())
    }

    fn update(&mut self, generate_cartoon: impl FnOnce(&mut Self) -> Option<Mat>) {
        self.old_cartoon = self.cartoon.clone();
        self.cartoon = generate_cartoon(self);
    }

    fn process(&mut self) -> Option<Mat> {
        self.gray_and_blur();
        self.adaptive_threshold();
        self.draw_contours();
        self.create_outline();
        self.quantize();
        self.cartoonify()
    }

}
