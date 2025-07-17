import yaml
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

class PromptManager:
    def __init__(self, prompts_dir: str, config_path: str):
        self.env = Environment(
            loader=FileSystemLoader(prompts_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.config = yaml.safe_load(Path(config_path).read_text())['prompts']
        self.cache = {}

    def get(self, key: str, **overrides) -> str:
        if key not in self.config:
            raise KeyError(f"Prompt key '{key}' not found in config.")
        entry = self.config[key]
        template_name = entry['template']

        template = self.cache.get(key)
        if template is None:
            template = self.env.get_template(template_name)
            self.cache[key] = template

        params = {**entry.get('defaults', {}), **overrides}
        return template.render(**params)

# Example usage
if __name__ == '__main__':
    base_dir = Path(r"C:\Users\sayee\Documents\UTN_Sem2\Deep_learning_for_digital_humanities_and_social_sciences\Project_Multimodality\Code\Multi3Hate\vlm\inference\prompts")
    cfg = base_dir / 'config.yml'

    pm = PromptManager(str(base_dir), str(cfg))
    print(pm.get('general', country='Germany'))
    # print(pm.get('india'))
    # print(pm.get('regionalInterpretation', country='Brazil'))