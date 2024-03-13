from checker import get_model_list

model_list = get_model_list()

for model_n in model_list:
    print(f"elif model_name == \"{model_n}\":")
    print(f"\treturn models.{model_n}(pretrained=True)")