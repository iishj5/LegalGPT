from google.generativeai import configure, list_models

configure(api_key="AIzaSyCVhxMSqepS_weVa2VFUM10tMM57HGFX6c")

models = list_models()
print([m.name for m in models])
