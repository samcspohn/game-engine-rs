// use std::collections::HashMap;

// use egui::{Ui, Pos2};
// use lazy_static::lazy::Lazy;

// fn m_menu<R>(ui: &mut Ui, enable: bool, popup_name: &str, add_contents: impl FnOnce(&mut Ui) -> R){
//     // let right_clicked = ui.input().pointer.button_clicked(egui::PointerButton::Secondary);
//     let popup_id = ui.auto_id_with(popup_name);
//     // if ui.input().pointer.button_clicked(egui::PointerButton::Secondary) {
//         //     println!("clicked right");
//         //     // let popup_id = ui.auto_id_with("inspector_popup");

//         // }
//     static mut POPUP_POS: Lazy<HashMap<String,Pos2>> = Lazy::new(|| {
//         HashMap::default()
//     });

//     if enable {
//         println!("here");
//         ui.memory().toggle_popup(popup_id);
//         let pos = ui.input().pointer.hover_pos().unwrap_or(Pos2 { x: 50., y: 50. });
//         unsafe { POPUP_POS.insert(popup_name.into(), pos); }
//     }
//     if ui.memory().is_popup_open(popup_id) {
//         let area_response = egui::Area::new(popup_id)
//             .order(egui::Order::Foreground)
//             .fixed_pos(unsafe { *POPUP_POS.get(popup_name.into()).unwrap() })
//             .constrain(true)
//             .show(ui.ctx(), |ui| {
//                 egui::Frame::popup(ui.style()).show(ui, |ui| {
//                     add_contents(ui);
//                 })
//             })
//             .response;

//         if ui.input().key_pressed(egui::Key::Escape) || area_response.clicked_elsewhere() {
//             // ui.memory().close_popup();
//             // unsafe { key = -1 }
//         }
//     }
// }
