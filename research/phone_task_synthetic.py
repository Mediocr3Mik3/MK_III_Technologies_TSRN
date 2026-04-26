"""
Phone Task Synthetic Dataset Generator
======================================
Generates ~20K high-quality phone-specific instruction examples for Stage-2 SFT.
"""
import random
import json
from pathlib import Path
from typing import List, Tuple

SYSTEM_PROMPT = (
    "You are a helpful phone assistant. You have access to: "
    "calendar, contacts, messages, reminders, web search, maps. "
    "Respond concisely. When a tool call is needed, output the exact tool name "
    "followed by JSON arguments."
)

NAMES = ["Mom","Dad","Sarah","Mike","Jessica","David","Emma","Chris","Lisa","Tom"]
PLACES = ["Starbucks","the grocery store","the airport","the dentist","the gym"]
ITEMS = ["milk","eggs","bread","bananas","coffee","toothpaste","chicken","rice"]
EVENTS = ["team meeting","doctor appointment","lunch with Sarah","gym session","project review"]
DAYS = ["today","tomorrow","Monday","Tuesday","Wednesday","Thursday","Friday","next week"]
TIMES = ["9:00 AM","10:30 AM","12:00 PM","2:00 PM","3:30 PM","5:00 PM","6:30 PM","8:00 PM"]

def _fmt(tool: str, action: str, **kwargs) -> str:
    return json.dumps({"tool": tool, "action": action, **kwargs}, ensure_ascii=False)

def _sample(n: int, templates: List[Tuple[str,str]], pools: dict) -> List[Tuple[str,str,str]]:
    out = []
    for _ in range(n):
        user_tpl, resp_tpl = random.choice(templates)
        user = user_tpl.format(**{k: random.choice(v) for k, v in pools.items()})
        assistant = resp_tpl.format(**{k: random.choice(v) for k, v in pools.items()})
        out.append((SYSTEM_PROMPT, user, assistant))
    return out

def gen_calendar(n: int) -> List[Tuple[str,str,str]]:
    return _sample(n, [
        ("Schedule a {event} for {day} at {time}",
         '{"tool":"calendar","action":"create","title":"{event}","date":"{day}","time":"{time}"}'),
        ("What meetings do I have {day}?", '{"tool":"calendar","action":"query","date":"{day}"}'),
        ("Cancel my {event}", '{"tool":"calendar","action":"delete","title":"{event}"}'),
        ("Remind me at {time} {day} to {event}",
         '{"tool":"reminders","action":"create","title":"{event}","datetime":"{day} {time}"}'),
    ], {"event": EVENTS, "day": DAYS, "time": TIMES})

def gen_contacts(n: int) -> List[Tuple[str,str,str]]:
    return _sample(n, [
        ("Call {name}", '{"tool":"call","contact":"{name}"}'),
        ("What's {name}'s phone number?", '{"tool":"contacts","action":"query","name":"{name}"}'),
        ("Text {name} that I'm running late",
         '{"tool":"messages","action":"send","to":"{name}","body":"Running late, be there soon."}'),
    ], {"name": NAMES})

def gen_messages(n: int) -> List[Tuple[str,str,str]]:
    bodies = ["I'm on my way","Thanks!","See you soon","Can you call me?","I'll be 10 minutes late"]
    out = []
    for _ in range(n):
        name = random.choice(NAMES)
        body = random.choice(bodies)
        tpl = random.choice([
            (f"Read my last text from {name}", f'{{"tool":"messages","action":"read_last","from":"{name}"}}'),
            (f"Send a text to {name}: {body}", f'{{"tool":"messages","action":"send","to":"{name}","body":"{body}"}}'),
            (f"Do I have any unread messages?", '{"tool":"messages","action":"count_unread"}'),
        ])
        out.append((SYSTEM_PROMPT, tpl[0], tpl[1]))
    return out

def gen_reminders(n: int) -> List[Tuple[str,str,str]]:
    return _sample(n, [
        ("Set a reminder to {event} {day} at {time}",
         '{"tool":"reminders","action":"create","title":"{event}","datetime":"{day} {time}"}'),
        ("What are my reminders for {day}?", '{"tool":"reminders","action":"query","date":"{day}"}'),
        ("Delete my {event} reminder", '{"tool":"reminders","action":"delete","title":"{event}"}'),
    ], {"event": EVENTS, "day": DAYS, "time": TIMES})

def gen_maps(n: int) -> List[Tuple[str,str,str]]:
    return _sample(n, [
        ("Directions to {place}", '{"tool":"maps","action":"directions","destination":"{place}"}'),
        ("How long to get to {place}?", '{"tool":"maps","action":"eta","destination":"{place}"}'),
        ("Find a gas station near {place}", '{"tool":"maps","action":"search_nearby","query":"gas station","near":"{place}"}'),
    ], {"place": PLACES})

def gen_search(n: int) -> List[Tuple[str,str,str]]:
    queries = ["weather","news","stock market","recipe for pancakes","NBA scores","best pizza nearby"]
    out = []
    for _ in range(n):
        q = random.choice(queries)
        out.append((SYSTEM_PROMPT, f"Search the web for {q}", f'{{"tool":"web_search","query":"{q}"}}'))
    return out

def gen_shopping(n: int) -> List[Tuple[str,str,str]]:
    out = []
    for _ in range(n):
        item = random.choice(ITEMS)
        out.append((SYSTEM_PROMPT, f"Add {item} to my shopping list", f'{{"tool":"shopping_list","action":"add","item":"{item}"}}'))
    return out

def gen_misc(n: int) -> List[Tuple[str,str,str]]:
    out = []
    for _ in range(n):
        tpl = random.choice([
            ("What's the weather today?", '{"tool":"weather","location":"current"}'),
            ("Set an alarm for 7 AM tomorrow", '{"tool":"alarm","action":"create","time":"7:00 AM","date":"tomorrow"}'),
            ("Play my workout playlist", '{"tool":"music","action":"play_playlist","name":"workout"}'),
            ("Turn on Bluetooth", '{"tool":"settings","action":"toggle","setting":"bluetooth","value":true}'),
            ("Take a note: buy milk on the way home", '{"tool":"notes","action":"create","body":"buy milk on the way home"}'),
            ("Call Mom back", '{"tool":"call","contact":"Mom"}'),
        ])
        out.append((SYSTEM_PROMPT, tpl[0], tpl[1]))
    return out

def generate_all(target_total: int = 20000, seed: int = 42) -> List[Tuple[str,str,str]]:
    random.seed(seed)
    per_cat = target_total // 8
    data = []
    data += gen_calendar(per_cat)
    data += gen_contacts(per_cat)
    data += gen_messages(per_cat)
    data += gen_reminders(per_cat)
    data += gen_maps(per_cat)
    data += gen_search(per_cat)
    data += gen_shopping(per_cat)
    data += gen_misc(per_cat)
    # Fill remainder
    while len(data) < target_total:
        data += gen_misc(1)
    random.shuffle(data)
    return data[:target_total]

def save_jsonl(path: str, data: List[Tuple[str,str,str]]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for system, user, assistant in data:
            f.write(json.dumps({"system": system, "user": user, "assistant": assistant}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    data = generate_all(20000)
    save_jsonl("data/phone_tasks_synthetic.jsonl", data)
    print(f"Generated {len(data)} phone-task examples -> data/phone_tasks_synthetic.jsonl")
