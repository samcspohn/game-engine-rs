use egui::Ui;
use lazy_static::lazy_static;
use spin::lock_api::Mutex;

lazy_static! {
    pub static ref DRAG_DROP_DATA: Mutex<String> = Mutex::new(String::from(""));
}



// stolen from egui demo
pub fn drag_source(ui: &mut Ui, id: egui::Id, drag_data: String, body: impl Fn(&mut Ui)) {
    let is_being_dragged = ui.memory().is_being_dragged(id);

    let response = ui.scope(&body).response;
    
    // Check for drags:
    let response = ui.interact(response.rect, id, egui::Sense::drag());
    if response.hovered() {
        // ui.output().cursor_icon = egui::CursorIcon::Grab;
    }
    if response.drag_started() {

        *DRAG_DROP_DATA.lock() = drag_data;
    }
    
    if is_being_dragged {
        ui.output().cursor_icon = egui::CursorIcon::Grabbing;

        // Paint the body to a new layer:
        let layer_id = egui::LayerId::new(egui::Order::Tooltip, id);
        let response = ui.with_layer_id(layer_id, body).response;

        // Now we move the visuals of the body to where the mouse is.
        // Normally you need to decide a location for a widget first,
        // because otherwise that widget cannot interact with the mouse.
        // However, a dragged component cannot be interacted with anyway
        // (anything with `Order::Tooltip` always gets an empty [`Response`])
        // So this is fine!

        if let Some(pointer_pos) = ui.ctx().pointer_interact_pos() {
            let delta = pointer_pos - response.rect.center();
            ui.ctx().translate_layer(layer_id, delta);
        }
    }
}

pub fn drop_target<R>(
    ui: &mut Ui,
    can_accept_what_is_being_dragged: bool,
    body: impl FnOnce(&mut Ui) -> R,
) -> egui::InnerResponse<R> {
    let is_being_dragged = ui.memory().is_anything_being_dragged();

    let margin = egui::Vec2::splat(4.0);

    let outer_rect_bounds = ui.available_rect_before_wrap();
    let inner_rect = outer_rect_bounds.shrink2(margin);
    let where_to_put_background = ui.painter().add(egui::Shape::Noop);
    let mut content_ui = ui.child_ui(inner_rect, *ui.layout());
    let ret = body(&mut content_ui);
    let outer_rect = egui::Rect::from_min_max(outer_rect_bounds.min, content_ui.min_rect().max + margin);
    let (rect, response) = ui.allocate_at_least(outer_rect.size(), egui::Sense::hover());

    let style = if is_being_dragged && can_accept_what_is_being_dragged && response.hovered() {
        ui.visuals().widgets.active
    } else {
        ui.visuals().widgets.inactive
    };

    let mut fill = style.bg_fill;
    let mut stroke = style.bg_stroke;
    if is_being_dragged && !can_accept_what_is_being_dragged {
        // gray out:
        fill = egui::color::tint_color_towards(fill, ui.visuals().window_fill());
        stroke.color = egui::color::tint_color_towards(stroke.color, ui.visuals().window_fill());
    }

    ui.painter().set(
        where_to_put_background,
        egui::epaint::RectShape {
            rounding: style.rounding,
            fill,
            stroke,
            rect,
        },
    );

    egui::InnerResponse::new(ret, response)
}
