import gensim.downloader as api

loaded_models = {}

def get_model(model_name: str):
    """
    Loads a pre-trained model from gensim-data, caching it in memory.
    """
    if model_name not in loaded_models:
        print(f"Loading model '{model_name}'... (this may take a while)")
        loaded_models[model_name] = api.load(model_name)
        print("Model loaded successfully.")
    return loaded_models[model_name] 