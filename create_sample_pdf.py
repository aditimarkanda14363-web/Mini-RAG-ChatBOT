"""
Script to generate a multi-page sample PDF for testing the RAG chatbot.
"""

from pathlib import Path
import fitz  # PyMuPDF


def create_sample_pdf(output_path: Path):
    doc = fitz.open()

    # Page 1: History of Artificial Intelligence
    p1 = doc.new_page()
    rect1 = fitz.Rect(50, 50, 550, 750)
    p1_text = """ARTIFICIAL INTELLIGENCE RESEARCH REPORT
Volume 1, Issue 4 — Foundations of Intelligent Systems

Page 1: Historical Genesis and Conceptual Foundations

The origins of Artificial Intelligence trace back to Alan Turing's seminal 1950 paper 'Computing Machinery and Intelligence', which introduced the Turing Test as an operational criterion for machine intelligence. Turing proposed that a computer could be considered intelligent if a human interrogator could not distinguish its responses from those of a human.

In 1956, the term 'Artificial Intelligence' was officially coined at the Dartmouth Summer Research Project on Artificial Intelligence, organized by John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon. The Dartmouth workshop established AI as an autonomous academic discipline with the premise that every aspect of learning can in principle be described so precisely that a machine can simulate it.

Early milestones included the Logic Theorist developed by Allen Newell, Herbert Simon, and Cliff Shaw in 1955, capable of proving mathematical theorems, followed by Joseph Weizenbaum's ELIZA in 1966, one of the earliest conversational agents demonstrating syntactic pattern matching.
"""
    p1.insert_textbox(rect1, p1_text, fontsize=11, fontname="helv")

    # Page 2: Machine Learning and Vector Embeddings
    p2 = doc.new_page()
    rect2 = fitz.Rect(50, 50, 550, 750)
    p2_text = """Page 2: Machine Learning Paradigms and Dense Vector Embeddings

Modern artificial intelligence relies predominantly on Machine Learning (ML), defined by Arthur Samuel in 1959 as the field of study that gives computers the ability to learn without being explicitly programmed.

The primary learning paradigms are:
1. Supervised Learning: Algorithms learn mapping functions from labeled input-output pairs (e.g., classification and regression).
2. Unsupervised Learning: Discovering inherent patterns, clusterings, or latent distributions in unlabeled data (e.g., K-Means clustering, PCA).
3. Reinforcement Learning: Agents maximize cumulative rewards through trial-and-error environmental interactions based on Markov Decision Processes.

Vector Embeddings:
A cornerstone of modern Natural Language Processing is the vector embedding representation. An embedding maps discrete words, sentences, or documents into continuous vector spaces of dimension d (typically d=384 or d=768). The sentence-transformers model 'all-MiniLM-L6-v2' maps sentences to a 384-dimensional dense vector space where semantic closeness translates directly to high cosine similarity.
"""
    p2.insert_textbox(rect2, p2_text, fontsize=11, fontname="helv")

    # Page 3: Neural Networks, Transformers, and Local LLMs
    p3 = doc.new_page()
    rect3 = fitz.Rect(50, 50, 550, 750)
    p3_text = """Page 3: Deep Neural Networks, Transformers, and Local LLM Inference

In 2017, Vaswani et al. published the landmark paper 'Attention Is All You Need', introducing the Transformer architecture. Transformers discarded recurrent and convolutional neural networks in favor of multi-head self-attention mechanisms, enabling massive parallelization across GPU clusters during pre-training.

The self-attention calculation computes:
Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V
where Q (queries), K (keys), and V (values) are linear projections of input token representations.

Local Large Language Model Deployment:
Modern advancements have democratized generative AI by allowing quantized foundation models (such as LLaMA-3.2, Mistral, and Gemma) to run entirely offline on standard consumer hardware using engines like Ollama. Through techniques like 4-bit and 8-bit weight quantization (GGUF format), high-parameter models achieve low memory footprints while preserving analytical reasoning capabilities.
"""
    p3.insert_textbox(rect3, p3_text, fontsize=11, fontname="helv")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()
    print(f"Sample PDF successfully generated at: {output_path}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "sample_documents" / "artificial_intelligence_overview.pdf"
    create_sample_pdf(out)
