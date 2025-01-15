use serde::Deserialize;

#[derive(Deserialize)]
pub struct Config {
    pub models: Models,
    pub documents: Documents,
}

#[derive(Deserialize)]
pub struct Models {
    pub embedding_model: String,
    pub rag_model: String,
}

#[derive(Deserialize)]
pub struct Documents {
    pub documents_dir: String,
    pub pdf1: String,
    pub pdf2: String,
}
