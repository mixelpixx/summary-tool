import os
import logging
from pathlib import Path
import gradio as gr
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure the logging level
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Create an OpenAI client instance
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def summarize_text(text: str, context: str = "", custom_instructions: str = "") -> str:
    """Summarizes text using the OpenAI GPT model with streaming and custom instructions."""
    try:
        system_message = "You are a helpful assistant skilled in summarizing text. Provide concise, informative summaries."
        if custom_instructions:
            system_message += f" {custom_instructions}"

        stream = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Context (if any): {context}\n\nAnalyze and summarize the following text: {text}"}
            ],
            stream=True,
            max_tokens=500
        )
        summary = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                summary += chunk.choices[0].delta.content
        return summary
    except Exception as e:
        logger.error(f"Error in summarize_text: {str(e)}")
        raise

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
    return chunks

def calculate_chunk_importance(chunks: List[str]) -> List[float]:
    vectorizer = TfidfVectorizer().fit_transform(chunks)
    similarity_matrix = cosine_similarity(vectorizer)
    centrality_scores = np.sum(similarity_matrix, axis=1)
    return centrality_scores.tolist()

def process_file(file_path: Path, output_dir: Path, chunk_size: int = 1000, overlap: int = 100, custom_instructions: str = "") -> Tuple[str, List[dict]]:
    """Processes a file, summarizing it chunk by chunk with parallel processing."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chunks = chunk_text(text, chunk_size, overlap)
    importance_scores = calculate_chunk_importance(chunks)
    
    with ThreadPoolExecutor() as executor:
        future_to_chunk = {executor.submit(summarize_text, chunk, custom_instructions=custom_instructions): (i, chunk) for i, chunk in enumerate(chunks)}
        summary_chunks = []
        
        for future in as_completed(future_to_chunk):
            chunk_index, original_chunk = future_to_chunk[future]
            try:
                summary = future.result()
                summary_chunks.append({
                    "index": chunk_index,
                    "original": original_chunk,
                    "summary": summary,
                    "importance": importance_scores[chunk_index]
                })
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_index}: {str(e)}")
    
    # Sort chunks by importance and select top 70%
    summary_chunks.sort(key=lambda x: x['importance'], reverse=True)
    selected_chunks = summary_chunks[:int(len(summary_chunks) * 0.7)]
    selected_chunks.sort(key=lambda x: x['index'])
    
    # Generate final summary
    combined_summary = " ".join([chunk['summary'] for chunk in selected_chunks])
    final_summary = summarize_text(combined_summary, context="This is a meta-summary of the most important parts of the document.", custom_instructions=custom_instructions)
    
    # Prepare detailed output
    detailed_output = [
        f"Chunk {i+1} (Importance: {chunk['importance']:.2f}):\n"
        f"Original: {chunk['original'][:100]}...\n"
        f"Summary: {chunk['summary']}\n"
        for i, chunk in enumerate(selected_chunks)
    ]
    
    # Save summaries to the specified output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file_path = output_dir / f"{file_path.stem}_summary{file_path.suffix}"
    detailed_file_path = output_dir / f"{file_path.stem}_detailed_summary{file_path.suffix}"
    
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        f.write(final_summary)
    
    with open(detailed_file_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(detailed_output))
    
    return final_summary, detailed_output

def summarize_files_interface(file_info, chunk_size: str, overlap: str, output_dir: str, custom_instructions: str):
    """
    Summarizes a file and handles input validation.
    """
    try:
        file_path = Path(file_info.name)
        output_dir = Path(output_dir) if output_dir else file_path.parent
        chunk_size = int(chunk_size) if chunk_size else 1000
        overlap = int(overlap) if overlap else 100
        
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            return "Invalid chunk size or overlap. Please enter positive integers with overlap less than chunk size."
        
        final_summary, detailed_output = process_file(file_path, output_dir, chunk_size, overlap, custom_instructions)
        
        summary_file_path = output_dir / f"{file_path.stem}_summary{file_path.suffix}"
        detailed_file_path = output_dir / f"{file_path.stem}_detailed_summary{file_path.suffix}"
        
        return f"Summary saved to {summary_file_path}\nDetailed summary saved to {detailed_file_path}\n\nFinal Summary:\n{final_summary}"
    
    except Exception as e:
        logger.error(f"Error in summarize_files_interface: {str(e)}")
        return f"An error occurred: {str(e)}"

# Gradio interface
interface = gr.Interface(
    fn=summarize_files_interface,
    inputs=[
        gr.File(label="Select a markdown or text file"),
        gr.Textbox(label="Chunk Size (words)", value="1000"),
        gr.Textbox(label="Overlap (words)", value="100"),
        gr.Textbox(label="Output Directory (leave blank for same as input file)"),
        gr.Textbox(label="Custom Instructions for AI", lines=5, placeholder="Enter any custom instructions for the AI here...")
    ],
    outputs="text",
    title="Advanced File Summarizer",
    description="Summarize markdown and text files with customizable chunk size, overlap, output location, and AI instructions."
)

if __name__ == "__main__":
    interface.launch()