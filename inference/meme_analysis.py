import os
import yaml
from pathlib import Path
from prompt_manager import PromptManager
from llm_inference_service import (
    AsyncOpenAIClientAdapter,
    LLMInferenceService,
    OpenAIClientAdapter,
    AllowedModelId,
)
from message_utils import prepare_multimodal_message


prompts_dir = Path(__file__).parent / "prompts"
cfg = prompts_dir / "config.yaml"

prompt_manager = PromptManager(str(prompts_dir), str(cfg))
sync_adapter = OpenAIClientAdapter()
async_adapter = AsyncOpenAIClientAdapter()
service = LLMInferenceService(sync_adapter, async_adapter)


# Prepare the input message
# 1. System message: prompt
# Build your system message
def analyze_meme(
    image_path: str,
    caption: str,
    country: str = "Germany",
    dialect: str = "Bavarian",
    persona: str = "a social activist",
) -> str:

    # now just one call to render all templates
    system_prompt = prompt_manager.get("general", country="Germany")
    # 2) Build the two multimodal message params entirely in-memory
    system_msg = prepare_multimodal_message(
        role="system", blocks=[{"text": system_prompt}]
    )

    # User message:  image + text
    user_msg = prepare_multimodal_message(
        role="user",
        blocks=[{"text": caption}, {"image_path": image_path}],
    )

    combined_msg = [system_msg, user_msg]

    adapter = OpenAIClientAdapter()
    llm_response = service.get_responses([AllowedModelId.GPT_4O], messages=combined_msg)
    return llm_response


if __name__ == "__main__":
    print(
        analyze_meme(
            r"C:\Users\sayee\UTN_Projects\Multi3Hate\data\memes\hi\Advicejew\37.jpg",
            "SEE THIS HAT <sep> THRIFT STORE",
        )
    )


# Send it to llm via llm_inference_service

# Store the response in csv
# ImageID | ModelID | PromptID | Response

# ImageID = foldername-filenamewithoutExtension
# PromptID =
