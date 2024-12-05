use egui::{Color32, Pos2, Stroke, Ui};
use egui_plot::{Line, Plot, PlotPoints};
use serde::{Deserialize, Serialize};
use splines::{Interpolation, Key, Spline};

use crate::engine::particles::shaders::cs::i;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gradient {
    spline: Spline<f32, f32>,
}

impl Gradient {
    pub fn new() -> Self {
        Self {
            spline: Spline::from_vec(vec![
                Key::new(0.0, 0.0, Interpolation::Bezier(0.5)),
                Key::new(1.0, 1.0, Interpolation::Bezier(0.5)),
            ]),
        }
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
                let v = self.spline.sample(i as f32 / 100.0).unwrap_or(0.0);
                // println!("{}: {}", i, v);
                if let Some(last) = last {
                    ui.painter().line_segment(
                        [
                            Pos2::new(rect.left() + (i - 1) as f32 / 99.0 * rect.width(), rect.top() + last * rect.height()),
                            Pos2::new(rect.left() + i as f32 / 99.0 * rect.width(), rect.top() + v * rect.height())
                        ],
                        Stroke::new(POINT_SIZE / 2.0, Color32::RED)
                    );
                }
                last = Some(v);
    
            });
            // ui.painter().line_segment([Pos2::new(rect.left(),rect.top()), Pos2::new(rect.right(), rect.bottom())], Stroke::new(4.0, Color32::BLUE));
            for (i, key) in self.spline.keys().iter().enumerate() {
                ui.painter().circle_filled(
                    egui::pos2(
                        rect.left() + key.t * rect.width(),
                        rect.top() + key.value * rect.height(),
                    ),
                    POINT_SIZE,
                    egui::Color32::RED,
                );
                let b = ui.rect_contains_pointer(egui::Rect::from_min_max(
                    egui::pos2(
                        rect.left() + key.t * rect.width() - POINT_SIZE / 2.,
                        rect.top() + key.value * rect.height() - POINT_SIZE / 2.,
                    ),
                    egui::pos2(
                        rect.left() + key.t * rect.width() + POINT_SIZE / 2.,
                        rect.top() + key.value * rect.height() + POINT_SIZE / 2.,
                    ),
                ));
                point_picker |= b;
                if b && ui.input(|i| i.pointer.primary_clicked()) {
                    unsafe {
                        SELECTED_POINT = Some(i);
                    }
                }
            }
        }

        if response.clicked() {
            if point_picker && unsafe { SELECTED_POINT.is_some() } {
                // Do nothing, point is already selected
            } else {
                self.spline.add(Key::new(
                    ((response.interact_pointer_pos().unwrap().x - rect.left()) / rect.width()),
                    (response.interact_pointer_pos().unwrap().y - rect.top()) / rect.height(),
                    Interpolation::Linear,
                ));
                unsafe {
                    SELECTED_POINT = Some(self.spline.keys().len() - 1);
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
        } else if response.drag_released() {
            unsafe { SELECTED_POINT = None }
            response.mark_changed();
        } else if response.dragged() {
            if let Some(index) = unsafe { SELECTED_POINT } {
                let key = self.spline.get_mut(index).unwrap();
                self.spline.remove(index);
                self.spline.add(Key::new(
                    ((response.interact_pointer_pos().unwrap().x - rect.left()) / rect.width())
                        .clamp(0.0, 1.0),
                    ((response.interact_pointer_pos().unwrap().y - rect.top()) / rect.height())
                        .clamp(0.0, 1.0),
                    Interpolation::Bezier(0.5),
                ));
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
