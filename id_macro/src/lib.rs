extern crate proc_macro;
extern crate syn;
#[macro_use]
extern crate quote;

use proc_macro::TokenStream;

#[proc_macro_derive(ID)]
pub fn ID(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    let gen = impl_ID(&ast);
    gen
}

fn impl_ID(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let name_str = format!("\"{}\"", name);
    let id = const_fnv1a_hash::fnv1a_hash_64(name_str.as_bytes(), None);
    let a = quote! {
        impl ID_trait for #name {
            const ID: u64 = #id;
        }
    };
    a.into()
}