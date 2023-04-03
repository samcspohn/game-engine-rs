use std::collections::BTreeMap;

pub struct ColorGradient {
    pub nodes: BTreeMap<i32, [f32; 4]>,
}

fn toggle_ui_compact(ui: &mut egui::Ui, on: &mut bool) -> egui::Response {
    let desired_size = ui.spacing().interact_size.y * egui::vec2(2.0, 1.0);
    let (rect, mut response) = ui.allocate_exact_size(desired_size, egui::Sense::click());
    if response.clicked() {
        *on = !*on;
        response.mark_changed();
    }
    response.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Checkbox, *on, ""));

    if ui.is_rect_visible(rect) {
        let how_on = ui.ctx().animate_bool(response.id, *on);
        let visuals = ui.style().interact_selectable(&response, *on);
        let rect = rect.expand(visuals.expansion);
        let radius = 0.5 * rect.height();
        ui.painter()
            .rect(rect, radius, visuals.bg_fill, visuals.bg_stroke);
        let circle_x = egui::lerp((rect.left() + radius)..=(rect.right() - radius), how_on);
        let center = egui::pos2(circle_x, rect.center().y);
        ui.painter()
            .circle(center, 0.75 * radius, visuals.bg_fill, visuals.fg_stroke);
    }

    response
}

impl ColorGradient {
    pub fn edit(&mut self, ui: &mut egui::Ui) -> egui::Response {
        let desired_size = ui.spacing().interact_size.y * egui::vec2(10.0, 1.0);
        let (rect, mut response) = ui.allocate_exact_size(desired_size, egui::Sense::click());
        // let clicked = response.clicked();
        // response.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Checkbox, *on, ""));

        let mut color_picker = false;
        let mut key = -1;
        if ui.is_rect_visible(rect) {
            let visuals = ui.style().interact(&response);
            ui.painter()
                .rect(rect, 0.0, visuals.bg_fill, visuals.bg_stroke);
            for (x, y) in &self.nodes {
                // println!("{}, {}, {}, {:?},{},{}", rect.left(), rect.right(), x, rect.center(), rect.width(), rect.left() + (*x as f32 / 100.) * rect.width());
                ui.painter().add(egui::Shape::convex_polygon(
                    vec![
                        egui::pos2(
                            rect.left() + (*x as f32 / 100.) * rect.width() - 5.,
                            rect.top(),
                        ), // top left
                        egui::pos2(
                            rect.left() + (*x as f32 / 100.) * rect.width() + 5.,
                            rect.top(),
                        ), // tp right
                        egui::pos2(
                            rect.left() + (*x as f32 / 100.) * rect.width(),
                            rect.top() + 7.,
                        ), // bottom tip
                    ],
                    egui::Color32::from_rgb(
                        (y[0] * 255.) as u8,
                        (y[1] * 255.) as u8,
                        (y[2] * 255.) as u8,
                    ),
                    egui::Stroke::new(
                        visuals.fg_stroke.width,
                        egui::Color32::from_rgb(255, 255, 255),
                    ),
                ));
                let b = ui.rect_contains_pointer(egui::Rect::from_min_max(
                    egui::pos2(
                        rect.left() + (*x as f32 / 100.) * rect.width() - 5.,
                        rect.top(),
                    ), // top left
                    egui::pos2(
                        rect.left() + (*x as f32 / 100.) * rect.width() + 5.,
                        rect.top() + 7.,
                    ), // bottom tip
                ));
                if b {
                    key = *x;
                }
                color_picker |= b;
                // ui.painter().arrow(
                //     egui::pos2(rect.left() + (*x as f32 / 100.) * rect.width(), rect.top()),
                //     egui::vec2(0.0, 10.0),
                //     egui::Stroke {
                //         width: 2.0,
                //         color: egui::Color32::from_rgb(
                //             (y[0] * 255.) as u8,
                //             (y[1] * 255.) as u8,
                //             (y[2] * 255.) as u8,
                //         ),
                //     },
                // );
            }
        }

        if response.clicked() {
            if color_picker {
                // let mut hsva = color_cache_get(ui.ctx(), *rgba);
                // let response = color_edit_button_hsva(ui, &mut hsva, alpha);
                // *rgba = Rgba::from(hsva);
                // color_cache_set(ui.ctx(), *rgba, hsva);
                // response
                ui.color_edit_button_rgba_premultiplied(self.nodes.get_mut(&key).unwrap());
            } else {
                println!(
                    "{}",
                    (rect.left() - response.interact_pointer_pos().unwrap().x) / rect.width()
                );
                self.nodes.insert(
                    (((response.interact_pointer_pos().unwrap().x - rect.left()) / rect.width())
                        * 100.) as i32,
                    [1.0, 1.0, 1.0, 1.0],
                );

                response.mark_changed();
            }
        }

        response
    }
}
