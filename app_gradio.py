import gradio as gr

from scripts.chat import (
    load_chatveritas,
    chat
)

bot = None


def load_model(choice):

    global bot

    use_lora = choice == "Qwen + ChatVeritas LoRA"

    bot = load_chatveritas(use_lora=use_lora)

    return f"✅ Loaded: {choice}"


def respond(message, history):

    global bot

    if bot is None:
        return "⚠️ Please load a model first."

    return chat(bot, message)


with gr.Blocks(title="ChatVeritas") as demo:

    gr.Markdown("# ChatVeritas")

    gr.Markdown(
        "Retrieval-Augmented Generation with optional LoRA fine-tuned model."
    )

    with gr.Row():

        model_choice = gr.Radio(
            choices=[
                "Base Qwen",
                "Qwen + ChatVeritas LoRA"
            ],
            value="Qwen + ChatVeritas LoRA",
            label="Model"
        )

        load_button = gr.Button("Load Model")

    status = gr.Textbox(
        label="Status",
        interactive=False
    )

    chatbot = gr.ChatInterface(
        fn=respond
    )

    load_button.click(
        fn=load_model,
        inputs=model_choice,
        outputs=status
    )


demo.launch()