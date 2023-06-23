extern crate proc_macro;
extern crate syn;
#[macro_use]
extern crate quote;

use proc_macro::TokenStream;

#[proc_macro_derive(ComponentID)]
pub fn component(input: TokenStream) -> TokenStream {
    // Construct a string representation of the type definition
    // let s = input.to_string();

    // Parse the string representation
    let ast = syn::parse(input).unwrap();

    // Build the impl
    let gen = impl_component(&ast);
    gen
    // Return the generated impl
    // gen.parse().unwrap()
}

fn impl_component(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let name_str = format!("\"{}\"", name);
    let id = const_fnv1a_hash::fnv1a_hash_64(name_str.as_bytes(), None);
    // let name_str_ident = syn::ExprLit:: (name_str);
    let a = quote! {
        impl _ComponentID for #name {
            // const ID: u64 = std::any::type_name::<Self>().hash();
                const ID: u64 = #id;
            // fn hello_world() {
            //     println!("Hello, World! My name is {}", stringify!(#name));
            // }
        }
    };
    a.into()
}



#[proc_macro_derive(AssetID)]
pub fn asset_id(input: TokenStream) -> TokenStream {
    // Construct a string representation of the type definition
    // let s = input.to_string();

    // Parse the string representation
    let ast = syn::parse(input).unwrap();

    // Build the impl
    let gen = impl_asset_id(&ast);
    gen
    // Return the generated impl
    // gen.parse().unwrap()
}

fn impl_asset_id(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let name_str = format!("\"{}\"", name);
    let id = const_fnv1a_hash::fnv1a_hash_64(name_str.as_bytes(), None);
    // let name_str_ident = syn::ExprLit:: (name_str);
    let a = quote! {
        impl _AssetID for #name {
            // const ID: u64 = std::any::type_name::<Self>().hash();
                const ID: u64 = #id;
            // fn hello_world() {
            //     println!("Hello, World! My name is {}", stringify!(#name));
            // }
        }
    };
    a.into()
}
