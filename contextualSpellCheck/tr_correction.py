import spacy
import contextualSpellCheck
import pandas as pd
import argparse
from tqdm import tqdm
from pandarallel import pandarallel
import multiprocessing
import spacy_transformers

# Load the pre-trained model
def load_model(model_name):
    print(f"Loading model: {model_name}")
    nlp = spacy.load(model_name)
    return nlp

# Add the spell checker to the spaCy pipeline
def add_contextual_spellchecker(nlp, config):
    print("Adding contextual spell checker to the pipeline")
    spell_checker = nlp.add_pipe(
        "contextual_spellchecker",
        config={"debug": config.debug, "max_edit_dist": config.max_edit_dist, "model_name":config.model_name},
        last=True
    )
    return spell_checker

# Define the function to correct text
def correct_with_contextual_spellcheck(text, nlp):
    doc = nlp(text)
    return doc._.outcome_spellCheck

# Main script
def main(args):  
    model_name_seg = args.model_name.split('/')[1].split('-')
    model_name = '-'.join(model_name_seg[:3])
    nlp = load_model(args.spacy_model)

    # Configure and add the spell checker to the pipeline
    spell_checker = add_contextual_spellchecker(nlp, args)

    df = pd.read_csv(args.data_path, encoding="UTF-8")
    tqdm.pandas(desc="Correcting rows", total=len(df))

    # Apply the spell checker to the DataFrame
    if args.parallerize:     
        pandarallel.initialize(nb_workers = multiprocessing.cpu_count(),progress_bar = True)
        df["corrected"] = df_['random_corrupted'].parallel_apply(lambda text: correct_with_contextual_spellcheck(text, nlp))
    else:
        df["corrected"] = df_['random_corrupted'].apply(lambda text: correct_with_contextual_spellcheck(text, nlp))

    df.to_csv("./test_" + model_name + ".csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contextual Spell Checker with spaCy")
    parser.add_argument("--spacy_model", type=str, default="tr_core_news_trf", help="The name of the tr spacy model.")
    parser.add_argument("--model_name", type=str, required=True, help="The name of the pre-trained model.")
    parser.add_argument("--max_edit_dist", type=int, default=3, help="The maximum edit distance for the spell checker.")
    parser.add_argument("--debug", type=bool, default=False, help="The debug option.") 
    parser.add_argument("--data_path", type=str, required=True, help="The path of test csv.") 
    parser. add_argument("--parallerize", type=bool, default=False, help = "The paralel processing option")
    args = parser.parse_args()
    main(args)


