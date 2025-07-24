import torch
import json
from utils.image_utils import dynamic_preprocess, build_transform, load_image
from PIL import Image

# Função para classificar o documento
def classify_document(reference_image_path, image_path, model, tokenizer):
    # Carregar e processar as imagens
    pixel_values1 = load_image(reference_image_path, max_num=12)
    pixel_values2 = load_image(image_path, max_num=12)
    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
    num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

    # Criar o prompt
    question = '''
                Image-1: <image>\nImage-2: <image>\n
                You are an AI assistant specialized in document analysis. Your task is to compare two company documents and assess their **visual similarity** based on their layout structure.

                **Instructions:**
                Analyze the two provided document images and measure their **visual similarity** based on:
                - **Shapes and Elements:** Compare the presence of graphical components, tables, sections, headers, and any other visual elements.
                - **Layout Consistency:** Evaluate the spatial arrangement of text blocks, margins, and alignments.
                - **Content Type:** Ensure that both documents contain similar types of content (e.g., tables, forms, paragraphs), regardless of specific wording.

                **Similarity Scoring:**
                Assign a **similarity score** between **0 and 100**, where:
                - **90-100** → **Nearly identical**: Documents have almost no visual differences.
                - **70-89** → **Highly similar**: Documents share the same structure with minor variations (e.g., small alignment changes).
                - **50-69** → **Moderately similar**: Key components remain, but there are noticeable structural differences.
                - **30-49** → **Weak similarity**: Some elements are shared, but the overall layout is significantly different.
                - **0-29** → **Completely different**: The documents do not share a recognizable visual structure.

                **Output Format:**
                Respond **only** with a JSON object structured as follows:
                ```json
                {
                    "similarity_score": <value between 0 and 100>,
                    "category": "<one of: Nearly Identical, Highly Similar, Moderately Similar, Weak Similarity, Completely Different>",
                    "justification": "Briefly explain the key visual similarities or differences detected."
                }
    '''

    # Configuração de geração
    generation_config = dict(max_new_tokens=1024, do_sample=True)

    # Chamar o modelo InternVL2_5-26B
    response, _ = model.chat(tokenizer, pixel_values, question, generation_config,
                             num_patches_list=num_patches_list, history=None, return_history=True)

    # Tentar extrair JSON da resposta
    try:
        response_json = json.loads(response)
    except json.JSONDecodeError:
        response_json = response

    return response_json
