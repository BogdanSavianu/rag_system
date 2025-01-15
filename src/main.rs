use anyhow::{Context, Result};
use pdf_extract::extract_text;
use rag_system_mini::utils::Config;
use rig::cli_chatbot::cli_chatbot;
use rig::embeddings::EmbeddingsBuilder;
use rig::providers::openai;
use rig::vector_store::in_memory_store::InMemoryVectorStore;
use rig::vector_store::VectorStore;
use std::path::Path;
use toml::from_str;

fn load_pdf_content<P: AsRef<Path>>(file_path: P) -> Result<String> {
    if !file_path.as_ref().exists() {
        anyhow::bail!("PDF file does not exist: {:?}", file_path.as_ref());
    }

    println!("Attempting to read PDF: {:?}", file_path.as_ref());
    extract_text(file_path.as_ref())
        .with_context(|| format!("Failed to extract text from PDF: {:?}", file_path.as_ref()))
}

#[tokio::main]
async fn main() -> Result<()> {
    let openai_client = openai::Client::from_env();
    let config = from_str::<Config>(&std::fs::read_to_string("Config.toml")?)?;

    let embedding_model = openai_client.embedding_model(&config.models.embedding_model);

    let mut vector_store = InMemoryVectorStore::default();

    let current_dir = std::env::current_dir()?;
    let documents_dir = current_dir.join(config.documents.documents_dir);

    let pdf1_path = documents_dir.join(&config.documents.pdf1);
    //let pdf2_path = documents_dir.join(config.documents.pdf2);

    println!("Documents directory: {:?}", documents_dir);
    println!("PDF1 path: {:?}", pdf1_path);
    //println!("PDF2 path: {:?}", pdf2_path);

    if !documents_dir.exists() {
        anyhow::bail!("Documents directory does not exist: {:?}", documents_dir);
    }

    let pdf1_content = load_pdf_content(&pdf1_path)?;
    //let pdf2_content = load_pdf_content(&pdf2_path)?;

    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .simple_document(config.documents.pdf1.as_str(), &pdf1_content)
        //.simple_document("Modern_Physics", &pdf2_content)
        .build()
        .await?;

    vector_store.add_documents(embeddings).await?;

    let rag_agent = openai_client.context_rag_agent(config.models.rag_model.as_str())
        .preamble("You are a helpful assistant that answers questions based on the given context from PDF documents.")
        .dynamic_context(1, vector_store.index(embedding_model))
        .build();

    cli_chatbot(rag_agent).await?;

    Ok(())
}
