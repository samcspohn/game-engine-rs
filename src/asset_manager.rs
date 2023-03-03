use std::{collections::HashMap, sync::Arc};

use crate::inspectable::Inspectable_;



pub trait Asset {

}


pub trait AssetManager<T: Inspectable_> {

}

pub trait AssetManagerBase {

}

pub struct AssetsManager {
    pub asset_managers: HashMap<String ,Arc<dyn AssetManagerBase>>
}

impl AssetsManager {
    pub fn get(&self, ext: &str) -> Option<Arc<dyn AssetManagerBase>> {
        if let Some(o) = self.asset_managers.get(ext) {
            Some(o.clone())
        } else {
            None
        }
    }
}