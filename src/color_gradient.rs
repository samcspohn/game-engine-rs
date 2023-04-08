use nalgebra_glm as glm;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

#[derive(Serialize, Deserialize)]
pub struct ColorGradient {
    pub nodes: HashMap<i32, (f32, [f32; 4])>,
    id_gen: i32,
}

impl Default for ColorGradient {
    fn default() -> Self {
        Self {
            nodes: HashMap::from_iter([(0, (0., [1., 1., 1., 1.]))]),
            id_gen: 1,
        }
    }
}

impl ColorGradient {
    pub fn new() -> Self {
        let mut a = Self {
            nodes: HashMap::new(),
            id_gen: 1,
        };
        a.nodes.insert(0, (0., [1., 1., 1., 1.]));
        a
    }
    pub fn to_color_array(&self) -> [[f32; 4]; 200] {
        let mut a = [[1.0; 4]; 200];
        let mut sorted = self
            .nodes
            .iter()
            .map(|(i, n)| (n.0, n.1))
            .collect::<Vec<(f32, [f32; 4])>>();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        if sorted.len() == 0 {
            return a;
        }
        if sorted.len() == 1 {
            return [sorted[0].1; 200];
        }
        for x in sorted.windows(2) {
            let b = (x[0].0 * 200.) as usize;
            let e = (x[1].0 * 200.) as usize;
            let mut start = glm::make_vec4(&x[0].1);
            let step = (glm::make_vec4(&x[1].1) - start) / (e - b) as f32;
            for i in b..e {
                a[i] = start.into();
                start += step;
            }
        }
        let last = sorted.last().unwrap();
        let b = (last.0 * 200.) as usize;
        let e = 200 as usize;
        let start = glm::make_vec4(&last.1);
        // let step = glm::make_vec4(&x[1].1) - start;
        for i in b..e {
            a[i] = start.into();
            // start += step;
        }
        a
    }
    pub fn edit(&mut self, ui: &mut egui::Ui) -> egui::Response {
        let desired_size = ui.spacing().interact_size.y * egui::vec2(10.0, 1.0);
        let (rect, mut response) = ui.allocate_exact_size(
            desired_size,
            egui::Sense {
                click: true,
                drag: true,
                focusable: true,
            },
        );
        let mut color_picker = false;
        static mut rgba_premul: [f32; 4] = [1., 1., 1., 1.];
        static mut key: i32 = -1;
        static mut hsva: egui::epaint::Hsva = egui::epaint::Hsva {
            h: 0.0,
            s: 1.0,
            v: 1.0,
            a: 1.0,
        };
        if ui.is_rect_visible(rect) {
            let visuals = ui.style().interact(&response);
            ui.painter()
                .rect(rect, 0.0, visuals.bg_fill, visuals.bg_stroke);
            for (x, y) in &mut self.nodes {
                ui.painter().add(egui::Shape::convex_polygon(
                    vec![
                        egui::pos2(rect.left() + y.0 * rect.width() - 6., rect.top()), // top left
                        egui::pos2(rect.left() + y.0 * rect.width() + 6., rect.top()), // top right
                        egui::pos2(rect.left() + y.0 * rect.width(), rect.top() + 9.), // bottom tip
                    ],
                    egui::Color32::from_rgb(
                        (y.1[0] * 255.) as u8,
                        (y.1[1] * 255.) as u8,
                        (y.1[2] * 255.) as u8,
                    ),
                    egui::Stroke::new(
                        visuals.fg_stroke.width / 2.,
                        egui::Color32::from_rgb(255, 255, 255),
                    ),
                ));
                let b = ui.rect_contains_pointer(egui::Rect::from_min_max(
                    egui::pos2(rect.left() + y.0 * rect.width() - 8., rect.top()), // top left
                    egui::pos2(rect.left() + y.0 * rect.width() + 8., rect.bottom()), // bottom tip
                ));
                color_picker |= b;
                if b && (ui.input().pointer.primary_clicked()
                    || ui.input().pointer.secondary_clicked())
                {
                    unsafe {
                        rgba_premul = y.1;
                        key = *x;
                        let rgba = egui::Rgba::from_rgba_premultiplied(
                            rgba_premul[0],
                            rgba_premul[1],
                            rgba_premul[2],
                            rgba_premul[3],
                        );

                        hsva = egui::epaint::Hsva::from(rgba);
                    }
                }
            }
        }

        let popup_id = ui.auto_id_with("popup");
        const COLOR_SLIDER_WIDTH: f32 = 210.0;
        if ui.memory().is_popup_open(popup_id) {
            let area_response = egui::Area::new(popup_id)
                .order(egui::Order::Foreground)
                .fixed_pos(egui::pos2(rect.left(), rect.bottom()))
                .constrain(true)
                .show(ui.ctx(), |ui| {
                    ui.spacing_mut().slider_width = COLOR_SLIDER_WIDTH;
                    egui::Frame::popup(ui.style()).show(ui, |ui| {
                        if egui::color_picker::color_picker_hsva_2d(
                            ui,
                            unsafe { &mut hsva },
                            egui::color_picker::Alpha::BlendOrAdditive,
                        ) {
                            let rgba = unsafe { egui::Rgba::from(hsva) };
                            unsafe {
                                rgba_premul = rgba.to_array();
                            }
                            if let Some(a) = unsafe { self.nodes.get_mut(&key) } {
                                unsafe {
                                    a.1 = rgba_premul;
                                }
                            }
                            // button_response.mark_changed();
                        }
                    });
                })
                .response;

            if ui.input().key_pressed(egui::Key::Escape) || area_response.clicked_elsewhere() {
                ui.memory().close_popup();
                unsafe { key = -1 }
            }
        }
        if response.clicked() {
            if color_picker && unsafe { self.nodes.contains_key(&key) } {
                ui.memory().toggle_popup(popup_id);
            } else {
                println!(
                    "{}",
                    (rect.left() - response.interact_pointer_pos().unwrap().x) / rect.width()
                );
                self.nodes.insert(
                    self.id_gen,
                    (
                        ((response.interact_pointer_pos().unwrap().x - rect.left()) / rect.width()),
                        [1.0, 1.0, 1.0, 1.0],
                    ),
                );
                unsafe {
                    key = self.id_gen;
                    rgba_premul = [1.0, 1.0, 1.0, 1.0];
                    let rgba = egui::Rgba::from_rgba_premultiplied(
                        rgba_premul[0],
                        rgba_premul[1],
                        rgba_premul[2],
                        rgba_premul[3],
                    );

                    hsva = egui::epaint::Hsva::from(rgba);
                }
                self.id_gen += 1;
                ui.memory().toggle_popup(popup_id);

                response.mark_changed();
            }
        } else if response.secondary_clicked() {
            if color_picker {
                // ui.memory().toggle_popup(popup_id);
                unsafe {
                    if let Some(_) = self.nodes.remove(&key) {}
                    if ui.memory().is_popup_open(popup_id) {
                        key = -1;
                    }
                }
            }
        } else if response.drag_released() {
            unsafe { key = -1 }
        } else if response.dragged() {
            if let Some(a) = unsafe { self.nodes.get_mut(&key) } {
                a.0 += response.drag_delta().x / rect.width();
                a.0 = a.0.clamp(0.0, 1.0);
            }
        }

        response
    }
}
