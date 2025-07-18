import os
import yaml
import csv
from pathlib import Path
from prompt_manager import PromptManager
from llm_inference_service import (
    AsyncOpenAIClientAdapter,
    LLMInferenceService,
    OpenAIClientAdapter,
    AllowedModelId,
)
from message_utils import prepare_multimodal_message

# Paths
prompts_dir = Path(__file__).parent / "prompts"
cfg_path    = prompts_dir / "config.yaml"

# Initialize
prompt_manager = PromptManager(str(prompts_dir), str(cfg_path))
sync_adapter   = OpenAIClientAdapter()
async_adapter  = AsyncOpenAIClientAdapter()
service        = LLMInferenceService(sync_adapter, async_adapter)

def generate_image_id(image_path: str) -> str:
    p = Path(image_path)
    folder    = p.parents[2].name
    subfolder = p.parents[1].name
    stem      = p.stem
    return f"{folder}-{subfolder}-{stem}"

def analyze_meme_category(
    image_path: str,
    caption:    str,
    category:   str,
    country:    str = "India",
    dialect:    str = "Tamilian",
    persona:    str = "a social activist",
) -> dict:
    """
    Run **all** prompts under `category`, returning a mapping
    of prompt_key -> LLM response text.
    """
    # 1) Load raw YAML so we know which sub-keys exist under the category
    with open(cfg_path, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)

    if category not in full_cfg["prompts"]:
        raise KeyError(f"Category '{category}' not found in config.yaml")

    subprompts = full_cfg["prompts"][category].keys()

    results = {}
    for prompt_key in subprompts:
        # 2) Build the system prompt for this sub-prompt
        composite_key = f"{category}.{prompt_key}"
        system_prompt = prompt_manager.get(
            composite_key,
            country=country,
            dialect=dialect,
            persona=persona,
        )
        

        system_msg = prepare_multimodal_message(
            role="system",
            blocks=[{"text": system_prompt}],
        )
        user_msg = prepare_multimodal_message(
            role="user",
            blocks=[{"text": caption}, {"image_path": image_path}],
        )
        combined = [system_msg, user_msg]

        # 3) Call the LLM
        model_enum   = AllowedModelId.GPT_4O
        llm_out      = service.get_responses([model_enum], messages=combined)
        results[prompt_key] = llm_out[model_enum].content

    return results

if __name__ == "__main__":
    image_path = r"C:\Users\sayee\UTN_Projects\Multi3Hate\data\memes\hi\American-Pride-Eagle\8.jpg"
    caption    = "I find lack of energy in this mail group <sep> very disturbing"
    category   = "CoreCulturalUnderstanding"   # <-- your category name here

    # 1) Analyze under each sub-prompt
    responses_by_key = analyze_meme_category(image_path, caption, category)

    # 2) Write each to CSV
    csv_path = "llm_responses.csv"
    file_exists = os.path.isfile(csv_path)
    header = ["ImageID", "ModelID", "PromptID", "Response"]

    with open(csv_path, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)

        image_id = generate_image_id(image_path)
        model_id = AllowedModelId.GPT_4O.value

        for prompt_key, response_text in responses_by_key.items():
            # Build a distinct PromptID: category.prompt_key
            prompt_id = f"{category}.{prompt_key}"
            writer.writerow([image_id, model_id, prompt_id, response_text])

    print(f"Saved {len(responses_by_key)} rows to {csv_path}")
