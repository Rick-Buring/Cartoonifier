use opencv::{
    core::Mat,
    imgcodecs::{imread, IMREAD_COLOR},
    imgproc::{self, AdaptiveThresholdTypes, ThresholdTypes},
    core::{MatTrait, MatTraitConst, Scalar, TermCriteria_EPS, TermCriteria_MAX_ITER},
    prelude::*
};

pub(crate) struct Cartoonify {
    pub original_image: Option<Mat>,
    pub old_cartoon: Option<Mat>,
    pub cartoon: Option<Mat>,

    //LoadImage
    pub(crate) path: String,
    //medianBlur
    pub(crate) blur_factor: i32,
    //adaptive blur
    pub(crate) max_value: f64,
    pub(crate) block_size: i32,
    pub(crate) c: f64,
    //Contours
    pub(crate) min_area: f64,
    pub(crate) min_length: f64,
    //dilation
    pub(crate) kernel_x: i32,
    pub(crate) kernel_y: i32,
    pub(crate) iterations: i32,
    //quantizing
    pub(crate) groups: i32,
    pub(crate) attempts: i32,
    pub(crate) max_count: i32,
    pub(crate) epsilon: f64,

    gray_blur: Mat,
    adaptive_threshold: Mat,
    contour_image: Mat,

    outline_image: Mat,
    quantize_image: Mat,
    combined_image: Mat
}

impl Cartoonify {
    pub fn set_path(&mut self, path: String) {
        self.path = path;
    }
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
    pub(crate) fn load_image(&mut self) -> Option<Mat> {
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

    pub(crate) fn update(&mut self, generate_cartoon: impl FnOnce(&mut Self) -> Option<Mat>) {
        self.old_cartoon = self.cartoon.clone();
        self.cartoon = generate_cartoon(self);
    }

    pub(crate) fn process(&mut self) -> Option<Mat> {
        if(self.original_image.is_none()){
            self.load_image();
        }
        self.gray_and_blur();
        self.adaptive_threshold();
        self.draw_contours();
        self.create_outline();
        self.quantize();
        self.cartoonify()
    }

}
