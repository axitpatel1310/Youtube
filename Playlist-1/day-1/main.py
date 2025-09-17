import json
import random
from typing import Dict, List
from utils import tokenize, clean_text

INTENTS_FILE = "faq.json"

def load_intents(path: str = INTENTS_FILE) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def match_intent_by_keyword(tokens: List[str], intents: Dict) -> str:
   
    best_tag = "fallback"
    best_score = 0

    for intent in intents["intents"]:
        tag = intent["tag"]
        patterns = intent.get("patterns", [])
        score = 0
        for p in patterns:
            p_clean = clean_text(p)
            p_tokens = p_clean.split()
            # count overlap
            for t in tokens:
                if t in p_tokens:
                    score += 1
        if score > best_score:
            best_score = score
            best_tag = tag

    return best_tag

def get_response_for_tag(tag: str, intents: Dict) -> str:
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            responses = intent.get("responses", [])
            if responses:
                return random.choice(responses)
    
    return "Sorry, I couldn't understand that."

def run_cli_bot():
    print("NEXTGEN AI")
    intents = load_intents()
    while True:
        user = input("You: ").strip()
        if not user:
            continue
        if user.lower() in ("quit", "exit"):
            print("Bot: Goodbye!")
            break
        tokens = tokenize(user)
        tag = match_intent_by_keyword(tokens, intents)
        response = get_response_for_tag(tag, intents)
        print("Bot:", response)

if __name__ == "__main__":
    run_cli_bot()
