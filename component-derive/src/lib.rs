extern crate proc_macro;
extern crate syn;
#[macro_use]
extern crate quote;

use proc_macro::TokenStream;

#[proc_macro_derive(ComponentID)]
pub fn component(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    let gen = impl_component(&ast);
    gen
}

fn impl_component(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let name_str = format!("\"{}\"", name);
    let id = const_fnv1a_hash::fnv1a_hash_64(name_str.as_bytes(), None);
    let a = quote! {
        impl _ComponentID for #name {
            const ID: u64 = #id;
        }
    };
    a.into()
}

#[proc_macro_derive(AssetID)]
pub fn asset_id(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    let gen = impl_asset_id(&ast);
    gen
}

fn impl_asset_id(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let name_str = format!("\"{}\"", name);
    let id = const_fnv1a_hash::fnv1a_hash_64(name_str.as_bytes(), None);
    let a = quote! {
        impl _AssetID for #name {
            const ID: u64 = #id;
        }
    };
    a.into()
}
