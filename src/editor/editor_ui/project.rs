use std::{
    collections::{HashMap, VecDeque},
    fs,
    path::{self, PathBuf},
    sync::Arc,
};

use egui::{Color32, Rect, Rounding, Sense, Ui};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use puffin_egui::puffin;

use crate::{
    editor::editor_ui::{
        entity_inspector::{self, _selected},
        GameObjectContextMenu, TransformDrag,
    },
    engine::{
        prelude::{Ins, Inspectable_, Sys, Transform},
        project::{asset_manager::AssetsManager, file_watcher::FileWatcher},
        utils,
        world::World,
    },
};

use super::EditorWindow;
pub(super) struct ProjectWindow {
    // file checker thread
}
impl ProjectWindow {
    pub fn new() -> ProjectWindow {
        ProjectWindow {}
    }
}
impl EditorWindow for ProjectWindow {
    fn draw(
        &mut self,
        ui: &mut Ui,
        editor_args: &mut super::EditorArgs,
        inspectable: &mut Option<Arc<Mutex<dyn Inspectable_>>>,
        rec: Rect,
        id: egui::Id,
    ) {
        let assets_manager = editor_args.assets_manager.clone();
        let file_watcher = &editor_args.file_watcher;
        project(
            ui,
            editor_args.world,
            inspectable,
            assets_manager,
            file_watcher,
            rec,
            id,
        )
    }

    fn get_name(&self) -> &str {
        "Project"
    }
}

pub(crate) fn project(
    ui: &mut egui::Ui,
    world: &mut World,
    inspectable: &mut Option<Arc<Mutex<dyn Inspectable_>>>,
    assets_manager: Arc<AssetsManager>,
    file_watcher: &FileWatcher,
    rec: Rect,
    id: egui::Id,
) {
    static mut INSPECTABLE: Option<Arc<Mutex<dyn Inspectable_>>> = None;
    unsafe { INSPECTABLE = inspectable.clone() };
    // let world = world.lock();
    use substring::Substring;
    let cur_dir: PathBuf = ".".into();
    fn render_dir(
        ui: &mut egui::Ui,
        cur_dir: PathBuf,
        sys: &Sys,
        assets_manager: &Arc<AssetsManager>,
    ) {
        // let label = format!("{:?}", cur_dir);
        let label: String = cur_dir.clone().into_os_string().into_string().unwrap();
        let id = ui.make_persistent_id(label.clone());
        let _cur_dir = cur_dir.clone();
        if let Ok(it) = fs::read_dir(cur_dir) {
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
                .show_header(ui, |ui| {
                    if let Some(last_slash) = label.rfind(path::MAIN_SEPARATOR_STR) {
                        let _label = label.substring(last_slash + 1, label.len());
                        let resp = ui.label(_label); //.sense(Sense::click());
                        let resp = ui.interact(resp.rect, id, Sense::click());
                        resp.context_menu(|ui| {
                            ui.menu_button("new particle template", |ui| {
                                static mut TEXT: String = String::new();

                                unsafe {
                                    let resp2 = ui.text_edit_singleline(&mut TEXT);
                                    if resp2.lost_focus()
                                        && ui.input(|i| i.key_pressed(egui::Key::Enter))
                                    {
                                        ui.close_menu();
                                        assets_manager
                                            .new_asset(format!("{label}/{TEXT}.ptem").as_str());
                                        TEXT = "".into();
                                    }
                                }
                            });
                            ui.menu_button("new light template", |ui| {
                                static mut TEXT: String = String::new();

                                unsafe {
                                    let resp2 = ui.text_edit_singleline(&mut TEXT);
                                    if resp2.lost_focus()
                                        && ui.input(|i| i.key_pressed(egui::Key::Enter))
                                    {
                                        ui.close_menu();
                                        assets_manager
                                            .new_asset(format!("{label}/{TEXT}.lgt").as_str());
                                        TEXT = "".into();
                                    }
                                }
                            });
                        });
                    }
                })
                .body(|ui| {
                    let paths: Vec<_> = it.map(|r| r.unwrap()).collect();
                    let mut dirs: Vec<_> = paths
                        .iter()
                        .filter(|x| fs::read_dir(x.path()).is_ok())
                        .collect();
                    let mut files: Vec<_> = paths
                        .iter()
                        .filter(|x| fs::read_dir(x.path()).is_err())
                        .collect();
                    dirs.sort_by_key(|dir| dir.path());
                    files.sort_by_key(|dir| dir.path());
                    for entry in dirs {
                        render_dir(ui, entry.path(), sys, assets_manager)
                    }
                    for entry in files {
                        render_dir(ui, entry.path(), sys, assets_manager)
                    }

                    // ui.label("The body is always custom");
                });
        } else {
            if let Some(last_slash) = label.rfind(path::MAIN_SEPARATOR_STR) {
                let _label: String = label.substring(last_slash + 1, label.len()).into();
                let item_id = egui::Id::new(label.clone());
                let path = utils::path_format(&_cur_dir);
                let resp = assets_manager.drag_source(ui, item_id, path.clone(), move |ui| {
                    ui.add(egui::Label::new(_label.clone()).sense(egui::Sense::click()));
                });
                if resp.clicked() {
                    unsafe { INSPECTABLE = assets_manager.inspect(&path) };
                }
                // }
            }
        }
    }
    render_dir(ui, cur_dir, &world.sys, &assets_manager);
    *inspectable = unsafe { INSPECTABLE.clone() };
}
