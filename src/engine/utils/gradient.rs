use egui::{Align2, Color32, FontId, Pos2, Stroke, Ui};
use egui_plot::{Line, Plot, PlotPoints};
use serde::{Deserialize, Serialize};
use splines::{Interpolate, Interpolation, Key, Spline};

use crate::engine::particles::shaders::cs::i;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gradient {
    spline: Spline<f32, MyType>,
    id_gen: usize,
}

// impl<(f32, usize)> Interpolate<(f32,usize) {
//     fn lerp(&self, other: &Self, t: f32) -> Self {
//         (self.0 * (1.0 - t) + other.0 * t, self.1)
//     }

//     fn step(t: T, threshold: T, a: Self, b: Self) -> Self {
//         todo!()
//     }

//     fn cosine(t: T, a: Self, b: Self) -> Self {
//         todo!()
//     }

//     fn cubic_hermite(t: T, x: (T, Self), a: (T, Self), b: (T, Self), y: (T, Self)) -> Self {
//         todo!()
//     }

//     fn quadratic_bezier(t: T, a: Self, u: Self, b: Self) -> Self {
//         todo!()
//     }

//     fn cubic_bezier(t: T, a: Self, u: Self, v: Self, b: Self) -> Self {
//         todo!()
//     }

//     fn cubic_bezier_mirrored(t: T, a: Self, u: Self, v: Self, b: Self) -> Self {
//         todo!()
//     }
// }
#[derive(Debug, Clone, Serialize, Deserialize, Copy)]
struct MyType(f32, usize);

impl Interpolate<f32> for MyType {
    fn step(t: f32, threshold: f32, a: Self, b: Self) -> Self {
        let a0 = a.0;
        let b0 = b.0;
        let result = f32::step(t, threshold, a0, b0);
        Self(result, a.1)
    }

    fn lerp(t: f32, a: Self, b: Self) -> Self {
        let result = a.0 * (1.0 - t) + b.0 * t;
        Self(result, a.1)
    }

    fn cosine(t: f32, a: Self, b: Self) -> Self {
        let result = (1.0 - (t * std::f32::consts::PI).cos()) / 2.0;
        Self(result, a.1)
    }

    fn cubic_hermite(
        t: f32,
        x: (f32, Self),
        a: (f32, Self),
        b: (f32, Self),
        y: (f32, Self),
    ) -> Self {
        let result = f32::cubic_hermite(
            t,
            (x.0, x.1 .0),
            (a.0, a.1 .0),
            (b.0, b.1 .0),
            (y.0, y.1 .0),
        );
        Self(result, a.1 .1)
    }

    fn quadratic_bezier(t: f32, a: Self, u: Self, b: Self) -> Self {
        let result = f32::quadratic_bezier(t, a.0, u.0, b.0);
        Self(result, a.1)
    }

    fn cubic_bezier(t: f32, a: Self, u: Self, v: Self, b: Self) -> Self {
        let result = f32::cubic_bezier(t, a.0, u.0, v.0, b.0);
        Self(result, a.1)
    }

    fn cubic_bezier_mirrored(t: f32, a: Self, u: Self, v: Self, b: Self) -> Self {
        let result = f32::cubic_bezier_mirrored(t, a.0, u.0, v.0, b.0);
        Self(result, a.1)
    }
}

impl Gradient {
    pub fn new() -> Self {
        Self {
            spline: Spline::from_vec(vec![
                Key::new(0.0, MyType(0.0, 0), Interpolation::Linear),
                Key::new(1.0, MyType(1.0, 1), Interpolation::Linear),
            ]),
            id_gen: 2,
        }
    }
    pub fn to_array(&self) -> [f32; 256] {
        let mut a = [0.0; 256];
        for i in (0..256) {
            // reverse as lifetime is 1.0 to 0.0
            a[i] = self
                .spline
                .clamped_sample((255 - i) as f32 / 255.0)
                .unwrap_or(MyType(0.0, 0))
                .0;
        }
        a
    }
    // pub fn add_color_stop(&mut self, t: f32, value: f32) {
    //     self.spline
    //         .add(Key::new(t, value, Interpolation::Linear));
    // }

    // pub fn remove_color_stop(&mut self, t: f32) {
    //     if let Some(index) = self.spline.keys().iter().position(|key| key.t == t) {
    //         self.spline.remove(index);
    //     }
    // }

    // pub fn get_value(&self, t: f32) -> Option<f32> {
    //     self.spline.sample(t)
    // }

    pub fn edit(&mut self, ui: &mut egui::Ui) -> egui::Response {
        const POINT_SIZE: f32 = 4.0;
        let desired_size = ui.spacing().interact_size.y * egui::vec2(10.0, 3.0);
        let (rect, mut response) = ui.allocate_exact_size(
            desired_size,
            egui::Sense {
                click: true,
                drag: true,
                focusable: true,
            },
        );
        // let points: PlotPoints = (0..256)
        //     .map(|i| {
        //         [
        //             i as f64 / 255.0,
        //             self.get_value(i as f32 / 255.0).unwrap_or(0.0) as f64,
        //         ]
        //     })
        //     .collect();
        // let line = Line::new(points);
        // Plot::new("Spline").show(ui, |plot_ui| {
        //     plot_ui.line(line);
        // });

        let mut point_picker = false;
        static mut SELECTED_POINT: Option<usize> = None;
        if ui.is_rect_visible(rect) {
            let visuals = ui.style().interact(&response);
            ui.painter()
                .rect(rect, 0.0, visuals.bg_fill, visuals.bg_stroke);
            let mut last = None;

            (0..100).for_each(|i| {
                // let v = self.get_value(i as f32 / 10.0).unwrap_or(0.0);
                let v = self
                    .spline
                    .clamped_sample(i as f32 / 100.0)
                    .unwrap_or(MyType(0.0, 0))
                    .0;
                // println!("{}: {}", i, v);
                if let Some(last) = last {
                    ui.painter().line_segment(
                        [
                            Pos2::new(
                                rect.left() + (i - 1) as f32 / 99.0 * rect.width(),
                                rect.bottom() - last * rect.height(),
                            ),
                            Pos2::new(
                                rect.left() + i as f32 / 99.0 * rect.width(),
                                rect.bottom() - v * rect.height(),
                            ),
                        ],
                        Stroke::new(POINT_SIZE / 2.0, Color32::RED),
                    );
                }
                last = Some(v);
            });
            ui.painter().line_segment(
                [
                    Pos2::new(rect.left(), rect.top()),
                    Pos2::new(rect.right(), rect.bottom()),
                ],
                Stroke::new(4.0, Color32::BLUE),
            );
            ui.painter().circle(
                Pos2::new(rect.left(), rect.top()),
                5.0,
                Color32::DARK_RED,
                Stroke::new(1.0, Color32::BLACK),
            );
            ui.painter().circle(
                Pos2::new(rect.right(), rect.bottom()),
                5.0,
                Color32::DARK_BLUE,
                Stroke::new(1.0, Color32::BLACK),
            );
            for key in self.spline.keys().iter() {
                ui.painter().circle_filled(
                    egui::pos2(
                        rect.left() + key.t * rect.width(),
                        rect.bottom() - key.value.0 * rect.height(),
                    ),
                    POINT_SIZE,
                    egui::Color32::RED,
                );
                ui.painter().text(
                    Pos2::new(
                        rect.left() + key.t * rect.width(),
                        rect.bottom() - key.value.0 * rect.height(),
                    ),
                    Align2::LEFT_TOP,
                    format!("{}", key.value.0),
                    FontId::monospace(11.0),
                    Color32::LIGHT_GRAY,
                );
                let b = ui.rect_contains_pointer(egui::Rect::from_min_max(
                    egui::pos2(
                        rect.left() + key.t * rect.width() - POINT_SIZE / 2.,
                        rect.bottom() - key.value.0 * rect.height() - POINT_SIZE / 2.,
                    ),
                    egui::pos2(
                        rect.left() + key.t * rect.width() + POINT_SIZE / 2.,
                        rect.bottom() - key.value.0 * rect.height() + POINT_SIZE / 2.,
                    ),
                ));
                point_picker |= b;
                if b && ui.input(|i| {
                    i.pointer.button_down(egui::PointerButton::Primary)
                        || i.pointer.button_down(egui::PointerButton::Secondary)
                }) {
                    unsafe {
                        SELECTED_POINT = Some(key.value.1);
                    }
                }
            }
        }

        if response.clicked() {
            if point_picker && unsafe { SELECTED_POINT.is_some() } {
                // Do nothing, point is already selected
            } else {
                let t = (response.interact_pointer_pos().unwrap().x - rect.left()) / rect.width();
                let id = self.id_gen;
                self.id_gen += 1;
                self.spline.add(Key::new(
                    t,
                    MyType(
                        (rect.bottom() - response.interact_pointer_pos().unwrap().y) / rect.height(),
                        id,
                    ),
                    Interpolation::Linear,
                ));
                unsafe {
                    // SELECTED_POINT = Some(id);
                    SELECTED_POINT = None;
                }
                response.mark_changed();
            }
        } else if response.secondary_clicked() {
            if point_picker {
                unsafe {
                    if let Some(index) = SELECTED_POINT {
                        self.spline.remove(index);
                        SELECTED_POINT = None;
                    }
                }
            }
        } else if response.drag_released()
            || ui.input(|i| i.pointer.button_released(egui::PointerButton::Primary))
        {
            unsafe { SELECTED_POINT = None }
            response.mark_changed();
        } else if response.dragged() {
            if let Some(selected) = unsafe { SELECTED_POINT } {
                let index = self
                    .spline
                    .keys()
                    .iter()
                    .position(|key| key.value.1 == selected)
                    .unwrap();
                let mut key = self.spline.remove(index).unwrap();
                let t = ((response.interact_pointer_pos().unwrap().x - rect.left()) / rect.width())
                    .clamp(0.0, 1.0);
                key.t = t;
                key.value.0 = ((rect.bottom() - response.interact_pointer_pos().unwrap().y)
                    / rect.height())
                .clamp(0.0, 1.0);
                self.spline.add(key);
                // unsafe {
                //     SELECTED_POINT = Some(key.value.1);
                // }
                // if let Some(key) = self.spline.get_mut(index) {
                //     key.t += response.drag_delta().x / rect.width();
                //     key.t = key.t.clamp(0.0, 1.0);
                //     key.value += response.drag_delta().y / rect.height();
                //     key.value = key.value.clamp(0.0, 1.0);
                // }
            }
        }

        response
    }
}
