🎹 Piano Prism: Stem Isolation for Music and OSTs Using MiniMax Audio Music-2.5

A sleek, AI-powered studio for generating OST music and cinematic visuals. This project leverages Minimax 2.5 Music and MiniMax-Hailuo 2.3 Video models, combined with HTDemucs for high-fidelity audio stem separation.

🚀 Features

    Fusion Music Generation: Creates complex compositions using Minimax 2.5.

    Cinematic Text-to-Video: Generates 1080p period-drama visuals using the latest MiniMax-Hailuo models.

    AI Stem Separation: Uses a U-Net Transformer approach (Demucs) to isolate piano tracks from the generated fusion audio.

    Comparison UI: A "Glassmorphism" Gradio interface to play original mixes vs. isolated stems side-by-side.

🛠️ Technical Stack
Component	Technology	Description
Generative Audio	Minimax 2.5 Music	Recently released model for high-fidelity 44.1kHz music.
Generative Video	Hailuo 2.3	State-of-the-art cinematic video generation at 1080p resolution.
Stem Separation	HTDemucs	Hybrid Transformer Demucs for professional-grade instrument isolation.
Interface	Gradio	Sleek, Python-based UI with custom "Soft" theme for hackathon demos.
Processing	FFmpeg	Complex filtergraphs for scaling 2.9K mobile recordings to 1080p standards.

📦 Installation & Usage

    Clone the Repository:

    bash
    git clone https://github.com/bortpro/piano_prism
    cd piano_prism

    Environment Setup:
    Add your MINIMAX_API_KEY to your environment variables or Google Colab Secrets.

    Run the Demo:
    Open piano_prism.ipynb in Google Colab and run all cells to launch the Gradio public link.



