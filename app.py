# --- 1. Import all necessary libraries and our custom modules ---
import gradio as gr
from analysis_engine import analyze_sentence
from visualizations import plot_attention_heatmaps, plot_embedding_space

# --- 2. Define the main controller function ---
# This function will be triggered by user actions (e.g., clicking a button).
# It takes inputs from the UI components and returns outputs for other UI components.
def process_and_visualize(model_name, sentence, layer_index):
    """
    Controller function that runs the analysis and generates all plots.
    """
    # Show a progress bar to the user
    progress = gr.Progress(track_tqdm=True)
    progress(0, desc="Model analiz ediliyor...")

    # Call our analysis engine from Phase 1
    analysis_data = analyze_sentence(model_name, sentence)

    if analysis_data is None:
        return None, None # Return None for plots if input is empty

    # Call our visualization functions from Phase 2
    progress(0.5, desc="Grafikler oluşturuluyor...")
    
    attention_fig = plot_attention_heatmaps(
        analysis_data["attention"],
        analysis_data["tokens"],
        int(layer_index) # Slider value can be float, ensure it's an integer
    )
    
    embedding_fig = plot_embedding_space(
        analysis_data["hidden_states"],
        analysis_data["tokens"]
    )

    # Return the generated figures to update the UI
    return attention_fig, embedding_fig


# --- 3. Build the UI using Gradio Blocks for a custom layout ---
with gr.Blocks(theme=gr.themes.Soft(), title="Transformer Gözlemcisi") as demo:
    gr.Markdown("# 🧠 Transformer Gözlemcisi")
    gr.Markdown("Bir Transformer modelinin içine girin ve bir cümlenin anlamını nasıl işlediğini görselleştirin.")

    with gr.Row():
        # --- LEFT COLUMN  ---
        with gr.Column(scale=1):
            gr.Markdown("### Ayarlar")
            model_dd = gr.Dropdown(
                label="Model Seçin",
                choices=["dbmdz/bert-base-turkish-cased", "bert-base-uncased", "distilbert-base-uncased"],
                value="dbmdz/bert-base-turkish-cased"
            )
            sentence_tb = gr.Textbox(
                label="Analiz Edilecek Cümle",
                value="Isparta, gülleri ve gölleri ile meşhur bir şehirdir.",
                lines=3
            )
            analyze_btn = gr.Button("Analizi Başlat", variant="primary")

        # --- RIGHT COLUMN ---
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("Dikkat Desenleri"):
                    gr.Markdown("Her bir 'dikkat kafasının' cümlenin anlamını oluştururken hangi kelimelere odaklandığını gösterir.")
                    layer_slider = gr.Slider(
                        label="İncelenecek Katman",
                        minimum=0,
                        maximum=11, # BERT-base has 12 layers (0-11)
                        step=1,
                        value=5
                    )
                    attention_plot = gr.Plot()

                with gr.TabItem("Kelime Gömülmeleri Uzayı"):
                    gr.Markdown("Modelin kelimeleri anlamsal olarak nasıl gruplandırdığını 2 boyutlu uzayda gösterir.")
                    embedding_plot = gr.Plot()
    
    # --- 4. Connect UI components to the controller function (Event Listeners) ---
    analyze_btn.click(
        fn=process_and_visualize,
        inputs=[model_dd, sentence_tb, layer_slider],
        outputs=[attention_plot, embedding_plot]
    )

# --- 5. Launch the Gradio app ---
if __name__ == "__main__":
    demo.launch()