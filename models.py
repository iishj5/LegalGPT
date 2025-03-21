from google.generativeai import configure, list_models

configure(api_key="API_KEY")

models = list_models()
print([m.name for m in models])
